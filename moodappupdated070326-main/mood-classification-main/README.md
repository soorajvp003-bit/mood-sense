# MoodSense Canonical App

This repository contains several older experiments, but the canonical runnable app is:

`moodapp-newfile-main\moodapp-newfile-main`

That integrated app now provides:

- text mood prediction
- webcam-based visual mood prediction
- single-face visual tracking that locks onto the current user
- SQLite-backed session storage
- RBAC-backed admin panel and user management
- weekly insights and support prompts

## Run

On Windows, use:

```bat
start_moodsense.bat
```

Then open:

`http://127.0.0.1:8000/`

## Notes

- Older folders such as `text`, `visual`, `text based`, and `visual_based` are legacy/training artifacts.
- The integrated backend serves its own UI and is the only app path that should be extended further.
