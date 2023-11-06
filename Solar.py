"""
Solar System simulation of the Sun and 11 orbital objects


Produces XYZ file of the planet positions
as function of time. Also produces file of
simulated data. 

Initial conditions taken: 2021-01-20
Vijay Chand
s1832195
"""

import sys
import numpy as np
import matplotlib.pyplot as pyplot
from Planet3D import Planet3D as P3D
import Observables as ob
import time

start = time.time()

G = 1.48818517e-34                                                              # Gravitational constant


def main():
    
    System_Parameters = sys.argv[1]                                             # All file inputs
    Initial_Conditions = sys.argv[2]
    XYZ_file = sys.argv[3]
    Observables_file = sys.argv[4]
    
    
    Total_Planets, dt, numstep, allowed_values = ob.Parameters(System_Parameters)      # Setting up simulation parameters
    time = 0.0
    count = 0                                                                   # Count used to mark data point in VMD file
    
    Observables_outfile = open( Observables_file, "w")                          # Opening VMD, simulation data file
    VMD_outfile = open( XYZ_file, "w")
    Planet_parameters = open( Initial_Conditions, "r")                          # Reading planet intial positions/mass

    Planets = []
    
    for i in range(Total_Planets):                                              # Creating list of planets as objects
    
        Planet = P3D.new_particle(Planet_parameters)
        Planets.append(Planet)
 
    Planets = P3D.Velocity_Correction(Planets, P3D.com_velocity(Planets))       # CoM velocity correction
    
    separation = ob.Planet_separation(Planets)                                  # Calculates initial planet seperations
    
    Initial_Potential_Energy = ob.Potential_energy(Planets, separation, G)      # Calculating initial kinetic, potential energies
    Initial_Kinetic_Energy = P3D.sys_kinetic(Planets)                           
    Energy = Initial_Kinetic_Energy + Initial_Potential_Energy
                                       
    Energy_Data = [Energy]                                                      # Initialising list data
    Kinetic_Energy = [Initial_Kinetic_Energy]
    Potential_Energy = [Initial_Potential_Energy]
    
    Time_Data = [time]
    Positional_Data = np.zeros((numstep,Total_Planets,3))
    
    for i in range(numstep):                                                    
        
        count += 1
        if count % allowed_values == 0:                                         # Allows the multiples of a user defined point to be printed into VMD file
            ob.OutPut_Format(Planets, VMD_outfile, count)                       # Prints planet positions to VMD file
        
        force_old = ob.Planet_Force(Planets,separation, G)                      # Calculates current forces on planets 
       
        P3D.update_pos_2nd_list(Planets, dt, force_old)                         # Calculates new positions of planets
        
        separation = ob.Planet_separation(Planets)                              # Calculates new planet seperations
        
        force_new = ob.Planet_Force(Planets,separation, G)                      # Calculates new forces on planets   
        
        P3D.update_vel_list(Planets, dt, 0.5*(force_old + force_new))           # Calculates new planet velocities          
        
        force_old = force_new                                                   # Sets old forces to new forces 
        
        time += dt                                                              # Calculates total time
        
        New_Kinetic_Energy = P3D.sys_kinetic(Planets)                           # Calculates new kinetic and potential energies for each position
        New_Potential_Energy = ob.Potential_energy(Planets, separation, G)
        Energy = New_Kinetic_Energy + New_Potential_Energy
    
        Kinetic_Energy.append(New_Kinetic_Energy)                               # Adds new energy and time data to appropriate list
        Potential_Energy.append(New_Potential_Energy)
        Energy_Data.append(Energy)
        Time_Data.append(time)
                
        
        for j in range(len(Planets)):
           
            Positional_Data[i,j] = Planets[j].position                          # Adding all planet positions to array

        

    Positional_Data = ob.Moon_Fix(Planets, Positional_Data)                     # Corrects moon positions wrt to the host planets frame
    
    
    pyplot.title('2D Simulation')                                               # Plots 2D Graphic of Simulation         
    x_comps = Positional_Data[:,:,0]                                            
    y_comps = Positional_Data[:,:,1]
    pyplot.xlabel('X - Position Components')
    pyplot.ylabel('Y - Position Components')
    pyplot.plot(x_comps, y_comps)
    pyplot.show()
    
    
    fig, ax = pyplot.subplots()                                                 # Plots Energy Flucuation vs Time Graph
    pyplot.title('Total Energy vs Time')                      
    pyplot.xlabel('Time')
    pyplot.ylabel('Energy Fluctuation')
    ax.plot(Time_Data, Energy_Data)
    pyplot.savefig("Energy Fluctuation Graph.png")                              # Saves plot to folder with code
    
    # Prints all data acquired from simulation to text file
    ob.Observables_File(Planets, sum(Kinetic_Energy), sum(Potential_Energy), Energy_Data, Positional_Data, Time_Data, Observables_outfile)
    
    Observables_outfile.close()                                                 # Closes simulated data and VMD file
    VMD_outfile.close()
    
main()

end = time.time()
print("{0} {1:.3f}s ".format("Program Run time =", end-start))                  # Prints simulation run time
