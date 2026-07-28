from atlasctf import Project

project = Project()
x = project.bitvec("x", 8)
project.require(f"{x.name} == 65")
print(project.solve().level)
