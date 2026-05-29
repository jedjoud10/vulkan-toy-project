# Mesh / Task Shader Notes
- On my system (win11), mesh shaders / task shaders will NOT work right after the computer boots up. When running the mesh/task shaders, the following will occur:
    1. the app will hang and drive will crash on the first attempt
    2. the system will BSOD with the `VIDEO_SCHEDULER_INTERNAL_ERROR` error code
- After rebooting post that BSOD, the task/mesh shaders will work.

Idk if this is because I am riding on UB somewhere in the task/mesh shader or if it's simply shit AMD drivers on windows.