#calculate reward
from random import randint

def chooseaction():
	return randint(1, 16)

def doaction(n):
	traci.trafficlight.setPhase("0", n)
	

	
def reward(list1,list2):
		s1=0
		s2=0
		n=len(list1)
		for i in range(0,n):
			s1 = s1 + list1[i]
		for j in range(0,n):
		    s2 = s2 + list2[j]
		return s1-s2

#convert states to number(encoding) 		
def conv(list1):
		n=len(list1)
		s=0

		for i in range(n-1,-1,-1):
			s=s+((3**(n-i-1))*list1[i])
		return s
print(conv([0,0,1,2]))


#helpgetstate takes the queue length and then returns state equivalent
def helpgetstate(m):
	if(m<3):
		return 0
	if(m<7):
		return 1
	return 2

#getstate returns state as a list
def getstate(n):

	l=[]
		
	for i in range(n):
		temp=helpgetstate(traci.inductionloop.getLastStepVehicleNumber(str(i))
		l.append(temp)

	return l
print(getstate(4))
