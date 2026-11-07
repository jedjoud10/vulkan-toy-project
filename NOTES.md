# Mesh / Task Shader Notes
- On my system (win11), mesh shaders / task shaders will NOT work right after the computer boots up. When running the mesh/task shaders, the following will occur:
    1. the app will hang and drive will crash on the first attempt
    2. the system will BSOD with the `VIDEO_SCHEDULER_INTERNAL_ERROR` error code
- After rebooting post that BSOD, the task/mesh shaders will work.

Idk if this is because I am riding on UB somewhere in the task/mesh shader or if it's simply shit AMD drivers on windows.

- Also IIRC (from yesterday lol) it crashes the driver if you try to read from the `groupshared` `vertices` or `triangles` memory in the mesh shader :P


# SDF
- For us to make use of the coarse cone-tracing pre-pass, all scene geometry must be defined in the global SDF. no local SDFs (or, at least, we store a bound in the global SDF)

- if we want to have soft SDF shadows, we must use global SDF. currently we use ray-traced shadows with hash offset to get soft shadows

- what if we do something similar to GPU work graphs, where we generate the SDF data (and cache it) on demand. all we would need is to "figure out" if a primitive needs to be generated if the ray intersects it, then generate it ASAP
- something something procedural streaming
