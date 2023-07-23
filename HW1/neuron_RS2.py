#   This MATLAB file generates figure 1 in the paper by 
#               Izhikevich E.M. (2004) 
#   Which Model to Use For Cortical Spiking Neurons? 
#   use MATLAB R13 or later. November 2003. San Diego, CA 

#   Modified by Ali Minai
#  and Ryan Connolly for HW1

############### regular spiking ######################
###############   QUESTION 1    ######################

import matplotlib.pyplot as plt
import matplotlib.axis as axs

steps = int(1000);                  #This simulation runs for 1000 steps

a=0.02; b=0.2; c=-65;  d=8;
V=-64; u=b * V;

tau = 0.25; 
ttspan = range(0, 4000, 1);  #tau is the discretization time-step
                                  #tspan is the simulation interval
# tspan = list(ttspan) / 4;
tspan = [x / 4 for x in ttspan];
RR_RS = [];
                                
T1=0;            #T1 is the time at which the step input rises
spike_ts = [];

VVV = [];
uuu = [];

for I in range(0, 41):
    VV=[];  uu=[];      # moved this here since I added VVV, holder of VVs
    R = 0;
    for tt in ttspan:
        t = tt / 4;
        # if (t > T1): 
        #     I=1.0;     # This is the input which you will change in your simulation
        # else:
        #     I=0;
        # I = 1; 

        V = V + tau * (0.04 * pow(V, 2) + 5 * V + 140 - u + I);
        u = u + tau * a * (b * V - u);
        if V > 30:                 #if this is a spike
            VV.append(30);         #VV is the time-series of membrane potentials
            V = c;
            u = u + d;
            spike_ts.append(1);   #records a spike
        else:
            VV.append(V);
            spike_ts.append(0);   #records no spike
        uu.append(u);

        if t > 200:
            R = R + spike_ts[len(spike_ts) - 1];
            

    VVV.append(VV);         # holds all the different I values' VV arrays for the plot
    uuu.append(uu);         # same but for u 
    RR_RS.append(R / 800); 


plt.subplot(5,1,1)                  # (rows, cols, which)
plt.plot(tspan, VVV[1]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
# axs.XAxis.set_label_coords(10, 10); 
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 1.1.1: Regular Spiking (I = 1)');

plt.subplot(5,1,2)                  # (rows, cols, which)
plt.plot(tspan, VVV[10]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 1.1.2: Regular Spiking (I = 10)');

plt.subplot(5,1,3)                  # (rows, cols, which)
plt.plot(tspan, VVV[20]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 1.1.3: Regular Spiking (I = 20)');

plt.subplot(5,1,4)                  # (rows, cols, which)
plt.plot(tspan, VVV[30]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 1.1.4: Regular Spiking (I = 30)');

plt.subplot(5,1,5)                  # (rows, cols, which)
plt.plot(tspan, VVV[40]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 1.1.5: Regular Spiking (I = 40)');

plt.figure();

plt.plot(list(range(0, 41)), RR_RS);
plt.axis([0, 40, 0, max(RR_RS)]);
plt.xlabel('I');
plt.ylabel('R');
plt.xticks([0, 40], [0, 40]);
plt.yticks([0, max(RR_RS)], [0, max(RR_RS)]); 
plt.title('Figure 1.2: R vs I');

plt.figure(); 

# plt.show()

# plt.subplot(2,1,2)
# plt.plot(tspan, spike_ts,'r');                   #spike train is plotted
# plt.axis([0, max(tspan), 0, 1.5])
# plt.xlabel('time step');
# plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
# plt.yticks([0, 1]);







############### fast spiking ######################
###############  QUESTION 2  ######################

steps = int(1000);                  #This simulation runs for 1000 steps

a=0.1; b=0.2; c=-65;  d=2;
V=-64; u=b * V;

tau = 0.25; 
ttspan = range(0, 4000, 1);  #tau is the discretization time-step
                                  #tspan is the simulation interval
# tspan = list(ttspan) / 4;
tspan = [x / 4 for x in ttspan];
RR_FS = [];
                                
T1=0;            #T1 is the time at which the step input rises
spike_ts = [];

VVV = [];
uuu = [];

for I in range(0, 41):
    VV=[];  uu=[];      # moved this here
    R = 0;
    for tt in ttspan:
        t = tt / 4;
        # if (t > T1): 
        #     I=1.0;     # This is the input which you will change in your simulation
        # else:
        #     I=0;
        # I = 1; 

        V = V + tau * (0.04 * pow(V, 2) + 5 * V + 140 - u + I);
        u = u + tau * a * (b * V - u);
        if V > 30:                 #if this is a spike
            VV.append(30);         #VV is the time-series of membrane potentials
            V = c;
            u = u + d;
            spike_ts.append(1);   #records a spike
        else:
            VV.append(V);
            spike_ts.append(0);   #records no spike
        uu.append(u);

        if t > 200:             # calculating total spike amount
            R = R + spike_ts[len(spike_ts) - 1];
            

    VVV.append(VV);
    uuu.append(uu);
    RR_FS.append(R / 800);      # using total spike amount to find average


plt.subplot(5,1,1)                  # (rows, cols, which)
plt.plot(tspan, VVV[1]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
# axs.XAxis.set_label_coords(10, 10); 
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 2.1.1: Fast Spiking (I = 1)');

plt.subplot(5,1,2)                  # (rows, cols, which)
plt.plot(tspan, VVV[10]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 2.1.2: Fast Spiking (I = 10)');

plt.subplot(5,1,3)                  # (rows, cols, which)
plt.plot(tspan, VVV[20]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 2.1.3: Fast Spiking (I = 20)');

plt.subplot(5,1,4)                  # (rows, cols, which)
plt.plot(tspan, VVV[30]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 2.1.4: Fast Spiking (I = 30)');

plt.subplot(5,1,5)                  # (rows, cols, which)
plt.plot(tspan, VVV[40]);                   #VV is plotted as the output
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 2.1.5: Fast Spiking (I = 40)');

plt.figure();

plt.plot(list(range(0, 41)), RR_RS, label = 'RS Neuron');   # Q1's R line
plt.plot(list(range(0, 41)), RR_FS, label = 'FS Neuron');   # Q2's R line
plt.legend(); 
plt.axis([0, 40, 0, max(max(RR_FS), max(RR_RS))]);
plt.xlabel('I');
plt.ylabel('R');
plt.xticks([0, 40], [0, 40]);
plt.yticks([0, max(max(RR_FS), max(RR_RS))], [0, max(max(RR_FS), max(RR_RS))]); 
plt.title('Figure 2.2: R vs I');

plt.figure(); 

#plt.show()




############### chattering (bursting) ######################
###############      QUESTION 3       ######################

steps = int(1000);                  #This simulation runs for 1000 steps

a=0.02; b=0.2; c=-50;  d=2;

VA = -65; uA = b * VA;      # need to separate them 
VB = -65; uB = b * VB; 

IA = 5.0
IB = 2.0


tau = 0.25; 
ttspan = range(0, 4000, 1);  #tau is the discretization time-step
                                  #tspan is the simulation interval
# tspan = list(ttspan) / 4;
tspan = [x / 4 for x in ttspan];
# RR_FS = [];
                                
T1=0;            #T1 is the time at which the step input rises
spike_ts_A = [];
spike_ts_B = [];

VVVA = []; uuuA = [];
VVVB = []; uuuB = [];

for W in range(0, 41):
    VVA = []; uuA = [];      # moved this here
    VVB = []; uuB = [];
    R = 0;
    IA_total = IA;
    IB_total = IB; 
    for tt in ttspan:
        t = tt / 4;

        
        # separated equations for neurons
        VA = VA + tau * (0.04 * pow(VA, 2) + 5 * VA + 140 - uA + IA);
        uA = uA + tau * a * (b * VA - uA);
        VB = VB + tau * (0.04 * pow(VB, 2) + 5 * VB + 140 - uB + IB);
        uB = uB + tau * a * (b * VB - uB);

        # if-statement for VA
        if VA > 30:                 #if this is a spike
            VVA.append(30);         #VV is the time-series of membrane potentials
            VA = c;
            uA = uA + d;
            spike_ts_A.append(1);   #records a spike
            IB_total = IB + W;
        else:
            VVA.append(VA);
            spike_ts_A.append(0);   #records no spike
            IB_total = IB; 

        # if-statement for VB
        if VB > 30:                 #if this is a spike
            VVB.append(30);         #VV is the time-series of membrane potentials
            VB = c;
            uB = uB + d;
            spike_ts_B.append(1);   #records a spike
            IA_total = IA - W;
        else:
            VVB.append(VB);
            spike_ts_B.append(0);   #records no spike
            IA_total = IA;
            
        uu.append(u);

##        if t > 200:
##            R = R + spike_ts[len(spike_ts) - 1];
            

    VVVA.append(VVA); uuuA.append(uuA);
    VVVB.append(VVB); uuuB.append(uuB); 
#    RR_FS.append(R / 800); 


plt.subplot(5,1,1)                      # (rows, cols, which)
plt.plot(tspan, VVVA[0], label = 'Neuron A');                   # VVA is plotted 
plt.plot(tspan, VVVB[0], label = 'Neuron B');                   # VVB is plotted
plt.legend();                                               # adds legend for color key
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
# axs.XAxis.set_label_coords(10, 10); 
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 3.1: Chattering (Bursting) (W = 0)');

plt.subplot(5,1,2)                      # (rows, cols, which)
plt.plot(tspan, VVVA[10], label = 'Neuron A');                   # VVA is plotted 
plt.plot(tspan, VVVB[10], label = 'Neuron B');                   # VVB is plotted
plt.legend();                                               # adds legend for color key
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
# axs.XAxis.set_label_coords(10, 10); 
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 3.2: Chattering (Bursting) (W = 10)');

plt.subplot(5,1,3)                      # (rows, cols, which)
plt.plot(tspan, VVVA[20], label = 'Neuron A');                   # VVA is plotted 
plt.plot(tspan, VVVB[20], label = 'Neuron B');                   # VVB is plotted
plt.legend();                                               # adds legend for color key
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
# axs.XAxis.set_label_coords(10, 10); 
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 3.3: Chattering (Bursting) (W = 20)');

plt.subplot(5,1,4)                      # (rows, cols, which)
plt.plot(tspan, VVVA[30], label = 'Neuron A');                   # VVA is plotted 
plt.plot(tspan, VVVB[30], label = 'Neuron B');                   # VVB is plotted
plt.legend();                                               # adds legend for color key
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
# axs.XAxis.set_label_coords(10, 10); 
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 3.4: Chattering (Bursting) (W = 30)');

plt.subplot(5,1,5)                      # (rows, cols, which)
plt.plot(tspan, VVVA[40], label = 'Neuron A');                   # VVA is plotted 
plt.plot(tspan, VVVB[40], label = 'Neuron B');                   # VVB is plotted
plt.legend();                                               # adds legend for color key
plt.axis([0, max(tspan), -90, 40])
plt.xlabel('time step');
# axs.XAxis.set_label_coords(10, 10); 
plt.ylabel('V_m');
plt.xticks([0, max(tspan)], [0, steps]);
# plt.xticklabels([0, steps]);
plt.title('Figure 3.5: Chattering (Bursting) (W = 40)');

#plt.figure();

##plt.plot(list(range(0, 41)), RR_RS, label = 'RS Neuron');   # Q1's R line
##plt.plot(list(range(0, 41)), RR_FS, label = 'FS Neuron');   # Q2's R line
##plt.legend(); 
##plt.axis([0, 40, 0, max(max(RR_FS), max(RR_RS))]);
##plt.xlabel('I');
##plt.ylabel('R');
##plt.xticks([0, 40], [0, 40]);
##plt.yticks([0, max(max(RR_FS), max(RR_RS))], [0, max(max(RR_FS), max(RR_RS))]); 
##plt.title('Figure 2.2: R vs I'); 

plt.show()




