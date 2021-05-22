import numpy as np 

a = [0]
b = [1]
c = [1,1,1,1,1]
d = [0,0,0,0,0]
e = [1,1,1,0,1,1,0,1,1]
f = [0,0,0,1,0,0,0]
g = [1,1,0,0,1,1,0]


def val(a):
	current = 0
	previous = 0
	maxSequence = 0
	for i in a:
		if i ==1:
			current+=1
		else:
			previous = current
			current = 0

		if previous+current > maxSequence:
			maxSequence = previous+current

	if maxSequence == len(a):
		maxSequence-=1
	print(maxSequence)

val(d)