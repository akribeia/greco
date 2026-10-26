/* Greco scheduler for independently optimized libLBFGS problems.
 * libLBFGS has Greco finite-trial/curvature safeguards; see README.md and
 * copyright.txt for provenance and its MIT license.
 * Linux ucontext fibers yield at evaluations, allowing one GPU batch without
 * sharing histories, step lengths, termination criteria or line searches.
 */
#include <ucontext.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lbfgs.h"

static void matmul(const double *a,const double *b,double *out) {
    for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
        double v=0.;for(int k=0;k<3;k++)v+=a[3*i+k]*b[3*k+j];out[3*i+j]=v;
    }
}
/* Double precision Rz Ry Rx, with the same analytic derivatives as the
 * independent NumPy reference. No fast-math or fused contraction. */
int greco_pose_geometry(const double *params,int count,const double *base,
                         const int *owner,int views,double *inverse,double *deriv,double *shift) {
    double *rot=malloc((size_t)count*36*sizeof(double));
    if(!rot)return -1;
    for(int p=0;p<count;p++) {
        const double *v=params+6*p;double sx=sin(v[0]),cx=cos(v[0]),sy=sin(v[1]),cy=cos(v[1]),sz=sin(v[2]),cz=cos(v[2]);
        double rx[9]={1,0,0,0,cx,-sx,0,sx,cx},ry[9]={cy,0,sy,0,1,0,-sy,0,cy},rz[9]={cz,-sz,0,sz,cz,0,0,0,1};
        double dx[9]={0,0,0,0,-sx,-cx,0,cx,-sx},dy[9]={-sy,0,cy,0,0,0,-cy,0,-sy},dz[9]={-sz,-cz,0,cz,-sz,0,0,0,0};
        double zy[9],tmp[9];double *r=rot+36*p;
        matmul(rz,ry,zy);matmul(zy,rx,r);matmul(zy,dx,r+9);
        matmul(rz,dy,tmp);matmul(tmp,rx,r+18);matmul(dz,ry,tmp);matmul(tmp,rx,r+27);
    }
    for(int o=0;o<views;o++) {
        const double *r=rot+36*owner[o],*b=base+9*o;
        for(int g=0;g<4;g++)for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
            double v=0.;for(int k=0;k<3;k++)v+=r[g*9+3*k+i]*b[3*k+j];
            if(g==0)inverse[o*9+3*i+j]=v;else deriv[o*27+(g-1)*9+3*i+j]=v;
        }
        for(int j=0;j<3;j++)shift[o*3+j]=params[owner[o]*6+3+j];
    }
    free(rot);
    return 0;
}

typedef struct batch batch;
typedef struct {
    batch *parent;
    ucontext_t context;
    void *stack;
    double *x, fx, response_fx, response_g[6];
    int pending, finished, status, iterations, evaluations;
} particle;
struct batch {
    ucontext_t scheduler;
    particle *particles;
    int count, abort, max_iterations, linesearch;
    double epsilon, xtol;
};

static double evaluate(void *instance,const double *x,double *g,int n,double step) {
    particle *p=instance;
    if(p->parent->abort){memset(g,0,n*sizeof(double));return 0.;}
    p->pending=1;p->evaluations++;
    swapcontext(&p->context,&p->parent->scheduler);
    if(p->parent->abort){memset(g,0,n*sizeof(double));return 0.;}
    memcpy(g,p->response_g,6*sizeof(double));
    return p->response_fx;
}
static int progress(void *instance,const double *x,const double *g,double fx,
                    double xnorm,double gnorm,double step,int n,int k,int ls) {
    particle *p=instance;p->iterations=k;
    return p->parent->abort;
}
static void run_particle(unsigned int lo,unsigned int hi) {
    uintptr_t address=(uintptr_t)lo|((uintptr_t)hi<<32);
    particle *p=(particle *)address;
    lbfgs_parameter_t options;lbfgs_parameter_init(&options);
    options.max_iterations=p->parent->max_iterations;
    options.epsilon=p->parent->epsilon;options.xtol=p->parent->xtol;
    options.linesearch=p->parent->linesearch;
    p->status=lbfgs(6,p->x,&p->fx,evaluate,progress,p,&options);
    p->pending=0;p->finished=1;
}
void greco_lbfgs_destroy(batch *b) {
    if(!b)return;
    b->abort=1;
    for(int i=0;i<b->count;i++) {
        particle *p=b->particles+i;
        if(p->stack && p->pending && !p->finished)
            swapcontext(&b->scheduler,&p->context);
        free(p->stack);lbfgs_free(p->x);
    }
    free(b->particles);free(b);
}
batch *greco_lbfgs_create(int count,int max_iterations,double epsilon,double xtol,int linesearch) {
    if(count<1 || max_iterations<1)return NULL;
    batch *b=calloc(1,sizeof(batch));if(!b)return NULL;
    b->count=count;b->max_iterations=max_iterations;b->epsilon=epsilon;b->xtol=xtol;
    b->linesearch=linesearch;
    b->particles=calloc(count,sizeof(particle));if(!b->particles){free(b);return NULL;}
    for(int i=0;i<count;i++) {
        particle *p=b->particles+i;p->parent=b;
        p->stack=malloc(256*1024);p->x=lbfgs_malloc(6);
        if(!p->stack || !p->x){greco_lbfgs_destroy(b);return NULL;}
        memset(p->x,0,6*sizeof(double));
        if(getcontext(&p->context)){greco_lbfgs_destroy(b);return NULL;}
        p->context.uc_stack.ss_sp=p->stack;p->context.uc_stack.ss_size=256*1024;
        p->context.uc_link=&b->scheduler;
        uintptr_t address=(uintptr_t)p;
        makecontext(&p->context,(void (*)(void))run_particle,2,
                    (unsigned int)address,(unsigned int)(address>>32));
        swapcontext(&b->scheduler,&p->context);
    }
    return b;
}
int greco_lbfgs_requests(batch *b,double *x,int *active) {
    int count=0;
    for(int i=0;i<b->count;i++) {
        particle *p=b->particles+i;memcpy(x+6*i,p->x,6*sizeof(double));
        active[i]=p->pending && !p->finished;count+=active[i];
    }
    return count;
}
void greco_lbfgs_advance(batch *b,const double *fx,const double *g) {
    for(int i=0;i<b->count;i++) {
        particle *p=b->particles+i;
        if(!p->pending || p->finished)continue;
        p->response_fx=fx[i];memcpy(p->response_g,g+6*i,6*sizeof(double));
        p->pending=0;swapcontext(&b->scheduler,&p->context);
    }
}
void greco_lbfgs_diagnostics(batch *b,int *status,int *iterations,int *evaluations) {
    for(int i=0;i<b->count;i++) {
        particle *p=b->particles+i;
        status[i]=p->status;iterations[i]=p->iterations;evaluations[i]=p->evaluations;
    }
}
