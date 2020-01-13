#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2017 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci


# ===============================================================================================

def chooseaction(num_actions):
    return random.randint(0, num_actions - 1)

def do_action(n,phno):
    ph = (phno + 1)%6 
    dur=(n+1)*5
    #traci.trafficlight.setPhase("0", ph)
    step = 0
    while step < dur:
        traci.trafficlight.setPhase("0", ph)
        traci.simulationStep()
        step = step + 1
    #for phase in range(

    
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
def encode_state(list1):
        n=len(list1)
        s=0

        for i in range(n-1,-1,-1):
            s=s+((3**(n-i-1))*list1[i])
        return s
#print(conv([0,0,1,2]))


#helpgetstate takes the queue length and then returns state equivalent
def helpgetstate(m):
    if(m<3):
        return 0
    if(m<7):
        return 1
    return 2

#getstate returns state as a list
def getstate(laneIDs):
    l=[]
    for i in laneIDs:
        #print('q = ',traci.lane.getLastStepHaltingNumber(i))
        temp=helpgetstate(traci.lane.getLastStepHaltingNumber(i))
        l.append(temp)
    return np.array(l)

def postgetstate(laneIDs,qlens):
    l=[]
    for i in laneIDs:
        #print('q = ',traci.lane.getLastStepHaltingNumber(i))
        t = traci.lane.getLastStepHaltingNumber(i)
        temp=helpgetstate(t)
        l.append(temp)
        qlens.append(t)
    return np.array(l)
# ===============================================================================================

'''
def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 50000  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        lastVeh = 0
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
        print("</routes>", file=routes)
'''

def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        lastVeh = 0
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>


def run():
    """execute the TraCI control loop"""
    tf.reset_default_graph()
    num_actions = 3
    qlens = []
    losses = []   #- changed 31st dec
    
#These lines establish the feed-forward part of the network used to choose actions
    inputs1 = tf.placeholder(shape=[1,81],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([81,num_actions],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1,num_actions],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    tf.summary.histogram("loss",loss)
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)


#   ======================================================================================================================== 
    init = tf.global_variables_initializer()
    #   ========================================================================================================================
    y = .99
    e = 0.1
    num_episodes = 4
    num_sigs = 4
    #create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                                 "--tripinfo-output", "tripinfo.xml"])
    detectorIDs = traci.inductionloop.getIDList()
    print('det ids - ', detectorIDs)
    laneIDs = []
    for det in detectorIDs:
        laneIDs.append(traci.inductionloop.getLaneID(det))
    print('lane ids - ', laneIDs)
    with tf.Session() as sess:
        #train_writer = tf.summary.FileWriter( 'logs/1/train ', sess.graph) - changed 31st dec
        sess.run(init)
        for i in range(num_episodes):
            #Reset environment and get first new observation
            #s = env.reset()
            #merge = tf.summary.merge_all()  - changed 31st dec
            print('epoch no - ',i)
            traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                                 "--tripinfo-output", "tripinfo.xml"])
            s_vec = np.array([0,0,0,0])
            s = 0
            rAll = 0
            d = False
            j = 0
            phno = -1
            losses_for_avg = []
            #The Q-Network
            while j < 3333:
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(81)[s:s+1]})
                if np.random.rand(1) < e:
                    a[0] = chooseaction(num_actions)
                    #a[0] = 0
                    #print('hi')
                #Get new state and reward from environment
                #s1,r,d,_ = env.step(a[0])
                #print('action = ',a[0])
                do_action(a[0],phno)
                s1_vec = getstate(laneIDs)
                #print(s1_vec)
                s1 = encode_state(s1_vec)
                r = reward(s_vec,s1_vec)
                #Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(81)[s1:s1+1]})
                #Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + y*maxQ1
                #Train our network using target and predicted Q values
                #summary,_,W1,los = sess.run([merge,updateModel,W,loss],feed_dict={inputs1:np.identity(81)[s:s+1],nextQ:targetQ}) - changed 31st dec
                _,W1,los = sess.run([updateModel,W,loss],feed_dict={inputs1:np.identity(81)[s:s+1],nextQ:targetQ})
                #los = sess.run(loss)
                #train_writer.add_summary(summary, j)  - changed 31st dec
                losses_for_avg.append(los) # changed 2 jan
                print("loss = ",los)
                rAll += r
                s = s1
                phno = (phno + 1) % 6
                if j==999:
                    #Reduce chance of random action as we train the model.
                    e = 1./((i/50) + 10)
                    break
            losses.append(sum(losses_for_avg)/len(losses_for_avg))  # changed 2 jan
            jList.append(j)
            rList.append(rAll)
            traci.close()
    #print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
        print("##################  TRAINING COMPLETE  ############################")
        print("##################  STARTING GREEDY APPLICATION  ##################")
    ####################################################################$$############################################################
        sumoBinary1 = checkBinary('sumo')
        traci.start([sumoBinary1, "-c", "data/cross.sumocfg",
                                 "--tripinfo-output", "tripinfo.xml"])
    #with tf.Session() as sess:
        #sess.run(init)
        step = 0    
        s_vec = np.array([0,0,0,0])
        s = 0
        phno = -1
        while step < 1000:
            step = step + 1
            #traci.simulationStep()
            a = sess.run(predict,feed_dict={inputs1:np.identity(81)[s:s+1]})
            #print('post action = ',a[0])
            do_action(a[0],phno)
            s1_vec = postgetstate(laneIDs,qlens)
            #print('post state - ',s1_vec)
            s1 = encode_state(s1_vec)
            s = s1
            phno = (phno + 1) % 6

        
        traci.close()
    ##################################################################$$############################################################
        print("##################  RL COMPLETE  ############################")
        print("##################  STARTING ROUND ROBIN  ##################")
        for du in [5,10,15]:
            traci.start([sumoBinary1, "-c", "data/roundrobin.sumocfg",
                                     "--tripinfo-output", "tripinfo.xml"])
            qlens_rr = []
            step = 0
            ph = -1
            while step < 500:
                step = step + 1
                ph = (ph + 1) % 6
                for i in range(du):
                    traci.trafficlight.setPhase("0", ph)
                    traci.simulationStep()
                
                if step % 5:
                    for lid in laneIDs:
                        qlens_rr.append(traci.lane.getLastStepHaltingNumber(lid))
                
            with open("result_test50.txt", "a") as myfile:
                myfile.write('\n AVG Q LEN WITH ROUNDROBINing - '+ str(sum(qlens_rr)/len(qlens_rr)) + ' for duration ' + str(du))            
            #print('AVG Q LEN WITH ROUNDROBIN - ',sum(qlens_rr)/len(qlens_rr),'for duration ',du)

        with open("result_test50.txt", "a") as myfile:
            myfile.write('\n AVG Q LEN WITH RL - ' + str(sum(qlens)/len(qlens)))
        #print('AVG Q LEN WITH RL - '+ str(sum(qlens)/len(qlens)))
    ####################################################################$$############################################################


    plt.plot(range(num_episodes), losses)
    plt.show()
    #traci.close()
    sys.stdout.flush()

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    #traci.start([sumoBinary, "-c", "data/cross.sumocfg","--tripinfo-output", "tripinfo.xml"])
    #traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                            # "--tripinfo-output", "tripinfo.xml"])
    run()
