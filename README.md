# Raytracer utilizing the HIP API from AMD

This is just a little project for me to learn about raytracing and GPU kernel programming.
I have generally accomplished the goals that I set out for myself when writing this, 
was to understand how these things work which I think I have a good grasp now. 

I will most likely not come back to this but if you are interested in building:

- Just git clone this repo
- cd into the folder
- make a build dir and cd into it
- cmake .. && make

Linux is only supported with this and it requires rocm and HIP libraries & runtime in order to work. In addition it utilizes
the C STB library to generate a PNG. Drop the folder into the libs dir and the build system should be able to find it.
