import random

# Generate 256 random float weights between -0.5 and 0.5
dummy_weights = [round(random.uniform(-0.5, 0.5), 4) for _ in range(256)]

with open("weights.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
    f.write("const int INPUT_SIZE = 256;\n")
    f.write("const float DUMMY_WEIGHTS[INPUT_SIZE] = {\n")
    
    # Format nicely with 8 weights per line
    for i in range(0, 256, 8):
        line = ", ".join(map(str, dummy_weights[i:i+8]))
        f.write(f"    {line},\n")
        
    f.write("};\n\n#endif\n")
print("weights.h generated successfully! Hand this over to Member 3.")

