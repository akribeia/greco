"""Explicit public frontend assets; paths outside this list are never served."""
SCRIPTS = ('core.js', 'projects.js', 'navigation.js', 'preprocessing.js', 'tilt-viewer.js', 'ctf-controls.js', 'reconstruction-steps.js', 'manual-curation.js', 'geometry.js', 'reconstruction.js', 'landmark-cloud.js', 'picking.js', 'manual.js', 'sta-cards.js', 'sta-viewer.js', 'sta-jobs.js', 'sta-actions.js', 'sta.js', 'segmentation.js', 'updates.js', 'app.js')
ROUTES = {'/': 'index.html', '/style.css': 'style.css',
          '/segmentation.css': 'segmentation.css', '/emblem.png': 'emblem.png'}
ROUTES.update(('/'+name, name) for name in SCRIPTS)
