"""
Observables module containing functions for  
integration method. Additionally, contains functions
for calculating observables data, i.e. period and apsides.

Vijay Chand
s1832195
"""

import numpy as np
import math

def Planet_separation(p3d_list):
    
    """
    Seperations between planets

    :param p3d_list: list of all planets/objects in the system
    :returns seperations between all planets: float np.array of position componenets
    """

    N = len(p3d_list)                                                           # Initialises seperations array and loop length
    separations = np.zeros([N,N,3])
    
    for i in range(N):
        for j in range(0,i):                                                    # Calculates half matrix of seperations of planets
            separations[i,j] = (p3d_list[i].position - p3d_list[j].position)    
            separations[j,i] = - separations[i,j]                               # Calculates remaining seperations through symmetry
            
    return separations 
    
    
    
def Potential_energy(p3d_list, separation, G):
        
    """
    Potential energy of the system

    :param p3d_list: list of all objects in the system
    :param G: gravitational constant
    :param seperation: matrix of seperation between planets
    :returns scalar potential energy of a list of planets/objects: float
    """

    N = len(p3d_list)
    radius = np.linalg.norm(separation, axis=2)                                 # Calculates the magnitude (radius) of all seperations
    Total_Grav_energy = [] 
 
    for i in range(N):
        for j in range(0,i):                                                    # Calculates potential energy for half the seperations
        
            if radius[i,j] != 0:                                                                  # Stops planet calculating potential energy wrt itself
                Planet_Grav_energy = - (G * p3d_list[i].mass * p3d_list[j].mass)/(radius[i,j])    # Calculates potential energy
            else:
                Planet_Grav_energy = 0
            
            Total_Grav_energy.append(Planet_Grav_energy)

    return sum(Total_Grav_energy)                                               # Returns total potential energy of the system



def Planet_Force(p3d_list, separation, G):
    
    """
    Resultant force on each planet

    :param p3d_list: list of all objects in the system
    :param G: gravitational constant
    :param seperation: matrix of seperation between planets
    :returns array of total resultant force on each planet: float np.array
    """
    
    N = len(p3d_list)
    radius = np.linalg.norm(separation, axis=2)                                 # Calculates the magnitude (radius) of all seperations 
    Force = np.zeros([N,N,3])                                                   # Initialises array for forces due to multiply planets
    Collective_Force = np.zeros([N,3])                                          # Initialises array for total resultant force on each planet
    
    for i in range(N):
        for j in range(0,i):                                                    # Calculates forces for half matrix of seperations
            
            if radius[i,j] != 0:                                                                                    # Stops planet calculating it's resultant force from itself
                Force[i,j] = ((G * p3d_list[i].mass * p3d_list[j].mass)/((radius[i,j] ** 3))) * separation[i,j]     # Calculates forces
                Force[j,i] = - Force[i,j]                                                                           # Calculates remaining forces through symmetry
            else:
                Force[i,j] = 0
                
    Collective_Force = - np.sum(Force, axis=1)                                  # Calculates the total resultant force acting on each planet
    
    return Collective_Force                                                     # Returns collective force on each planet



def OutPut_Format(p3d_list, filename, count):

    """
    Prints positions of all planets into file for a single iteration

    :param p3d_list: list of all objects in the system
    :param filename: name of output file for VMD
    :param count: iteration value corresponding to positional dataset
    """    

    output_string = str(len(p3d_list)) + "\n" + "Point = " + str(count) +"\n"   # Formats the initial 2 lines of VMD file, i.e. total planets and point number

    for i in p3d_list:
        
        output_string += str(i) + "\n"                                          # Formats all data for planets in VMD file
        
    filename.write(output_string)                                               # Writes formated data into VMD file
        
    
    
def Period(Positional_Data, Time_Data, p3d_list):
    
    """
    Calculates the orbital periods of all planets

    :param p3d_list: list of all objects in the system
    :param Positional_Data: Positional Data of all planets
    :param Time_Data: All recorded time data over the simulation
    :returns array of periods for all planets: float np.array
    """  
    
    Planet_Periods = np.zeros([len(p3d_list)-1])                                # Initialises planet period and angle arrays
    dTheta_list = np.zeros([len(Positional_Data)])
    
    for j in range(1,len(p3d_list)):                                            # Loops over all planets for all positions, starting at 1 to avoid Sun
        for i in range(len(Positional_Data)-1):
            
            a = Positional_Data[i,j]                                            # defines vector - (a) at intial position
            b = Positional_Data[i+1,j]                                          # defines successive vector - (b), this is located at next position after (a)
            
            a_mag = np.linalg.norm(a)                                           # calculates magnitude of vectors (a) and (b)
            b_mag = np.linalg.norm(b)
            
            dTheta = np.arccos(((np.dot(a,b))/(abs(a_mag) * abs(b_mag))))       # Calculates angle between them and adds to theta array
            dTheta_list[i] = dTheta                             
            
        dTheta_sum = sum(dTheta_list) 
        Rotations = dTheta_sum/(2 * math.pi)                                    # Calculates number of rotation by dividing sum of angles by 2pi
        Period = Time_Data[-1]/ Rotations                                       # Calculates period by dividing total time by number of rotations
        Planet_Periods[j-1] = Period                                            # Adds planet period to period array, its j-1 as the loop starts at 1
        
    return Planet_Periods                                                       # Returns all planet periods



def Apsides(Positional_Data, p3d_list):
    
    """
    Calculates apsides of all planets

    :param p3d_list: list of all objects in the system
    :param Positional_Data: Positional Data of all planets
    :returns array of apsides for all planets: float np.array (No. of Planets x 2) sized
    """
    
    Radius = np.zeros([len(p3d_list)-1, len(Positional_Data)])                  # Initialises radius and apside arrays for planets
    Aspides_list = np.zeros([len(p3d_list)-1,2]) 
     
    for i in range(1,len(p3d_list)):                                            # Loops over all planets and positional data, starting at 1 to avoid the Sun
        for j in range(len(Positional_Data)):
            
            if p3d_list[i].label == 'Moon':                                     # Condition to calculate the correct moon obrit wrt to the moons host planet
                 radius_components = Positional_Data[j,i]                       # This works becasue moons positional data is corrected before this function using Moon_fix function in Solar.py
                
            else:
                radius_components = Positional_Data[j,0] - Positional_Data[j,i]         # Calculates the radius of all planets except moons
            
            Radius[i-1,j] = np.linalg.norm(radius_components)                   # Calculates and adds radius to radius array, i-1 is used as the initial loop starts at 1
        
        Max = np.max(Radius[i-1,:])                                             # Finding the max and min apside radius for one planet
        Min = np.min(Radius[i-1,:])
        
        Aspides_list[i-1,0] = Max                                               # Adding the max and min apsides to apside array
        Aspides_list[i-1,1] = Min
        
    return Aspides_list                                                         # Returns total apsides for all planets and moons



def Parameters(file_handle):
    
    """
    Reads input parameters from file

    :param file_handle: input file which includes simulation parameters
    :returns parameters: Total_Planets, dt, numer of steps, allowed points to be printed
    """
    
    filename = open(file_handle, "r" )                                          # Opens file and reads parameters 
    
    file = filename
    line2 = file.readlines()[1].split()                                         # Reads the second line of System Parameters file
     
    Total_Planets = int(line2[0])                                               # Initialises all parameters
    dt = float(line2[1])                                                   
    numstep = int(line2[2])                         
    allowed_values = int(line2[3])
    
    return Total_Planets, dt, numstep, allowed_values                           # Returns all parameters 


            
def Moon_Fix(p3d_list, Positional_Data):
      
    """
    Corrects the moons positional data wrt to host planet

    :param p3d_list: list of all objects in the system
    :param Positional_Data: Positional Data of all planets
    :returns All planets positional data with moon corrections: float np.array 
    """
    
    for i in range(len(p3d_list)):                                              # Loops over all objects for all positional data
        for j in range(len(Positional_Data)):
            
            if p3d_list[i].label == 'Moon':                                     # If the objects has the name 'Moon' the correction begins
                
                Moon_Pos = Positional_Data[j,i]                                 
                Home_Pos = Positional_Data[j,i-1]                               # The planet which the moon orbits is always considered to be the one before the moon 
                Positional_Data[j,i] = Moon_Pos - Home_Pos                      # The correction is given by the positional data of the moon subtracted from the host planet 
    
    return Positional_Data                                                      # Returns all positional data with moon corrections
 
    
 
def Observables_File(p3d_list, Total_Kinetic_Energy, Total_Potential_Energy, Energy_Data, Positional_Data, Time_Data, filename):
    
    """
    Prints all simulation generated data to a file

    :param p3d_list: list of all objects in the system, list
    :param Total_Kinetic_Energy: Total kinetic energy in the simulation, float
    :param Total_Potential_Energy: Total potential energy in the simulation, float
    :param Energy_Data: All energy data in the simulation, list
    :param Positional_Data: Positional data for all planets, array
    :param Time_Data: All time data over the simulation, array
    :param filename: name of output file for calculated data
    :returns File containing planet apsides, periods, total kinetic and potential energy,
    total energy, energy fluctuation and simulation length in years.
    """
    
    apsides = Apsides(Positional_Data, p3d_list)                                # Calculates all apsides of planets/objects
    period = Period(Positional_Data, Time_Data, p3d_list)                       # Calculates all periods of objects 
    deltaE = abs(max(Energy_Data) - min(Energy_Data))/abs(Energy_Data[0])       # Calculates energy flucuation
    
    output_string = "{0:16s} {1:12s} {2:12s} {3:12s}\n".format("Planets","Apside_Max","Apside_Min","Period")              # Formats titles for the outfile

    
    for i in range(len(p3d_list)-1):
  
        output_string += "{0:15s} {1:12.5e} {2:12.5e} {3:12.5e}\n".format(p3d_list[i+1].label, apsides[i,0], apsides[i,1], period[i])           # Formats all periods and apside for outfile
    
    output_string += "\n" + "Global System Properties:\n"                                                                                                                           # Titles for solar system properties 
    output_string += "{0:16s} {1:18s} {2:16s} {3:20s} {4:18s}\n".format("Kinetic_Energy","Potential_Energy","Total_Energy","Energy_Fluctuation", "Simulation Length (Years)")
    output_string += "{0:12.7e} {1:17.7e} {2:18.7e} {3:14.6e} {4:15.3f}\n".format(Total_Kinetic_Energy, Total_Potential_Energy, sum(Energy_Data), deltaE, Time_Data[-1]/365)        # Formats system properties for file 
    
    filename.write(output_string)