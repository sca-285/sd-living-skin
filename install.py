import launch

# facexlib (face detection, face parsing, person matting) ships with most
# Forge / A1111 installs for face restoration; installed here if missing.
if not launch.is_installed("facexlib"):
    launch.run_pip("install facexlib", "Living Skin requirement: facexlib")
