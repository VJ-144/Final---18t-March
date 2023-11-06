"""
Planet3D, a class to describe planet properties in 3D space

An instance describes a planet in Euclidean 3D space: 
velocity and position are [3] arrays

Includes time integrator methods, linear momentum, kinetic energy,
1st & 2nd position updates and velocity update

Author: Vijay Chand
s1832195
"""

import numpy as np

class Planet3D(object):
    
    """
    Class to describe planet properties in 3D space

        Properties:
    label: name of the planet
    mass: mass of the planet
    pos: position of the planet
    vel: velocity of the planet

        Methods:
    __init__
    __str__
    kinetic_e  - computes the kinetic energy
    momentum - computes the linear momentum
    update_pos_1st - updates the position to 1st order
    update_pos_2nd - updates the position to 2nd order
    update_vel - updates the velocity

        Static Methods:
    new_particle - initializes a P3D instance from a file handle
    sys_kinetic - computes total K.E. of a p3d list
    com_velocity - computes total mass and CoM velocity of a p3d list
    update_pos_2nd_list - computes total 2nd order position updates of a p3d list
    update_vel_list - computes total velocity updates of a p3d list
    Velocity_Correction - computes velocity corrections for a p3d list
    """

    def __init__(self, label, mass, position, velocity):
        
        """
        Initialises a Planet in 3D space

        :param label: String w/ the name of the planet
        :param mass: float, mass of the planet
        :param position: [3] float array w/ position
        :param velocity: [3] float array w/ velocity
        """
        
        self.mass = float(mass)                                                 #Creating particle properties 
        self.position = position
        self.velocity = velocity
        self.label = str(label)



    @staticmethod
    def new_particle(file_handle):
        
        """
        Initialises a Planet3D instance given an input file handle.
        
        The input file should contain one planet per line in the following format:
        label   <mass>  <x> <y> <z>    <vx> <vy> <vz>
        
        :param input file: Readable file handle in the above format
        :return Planet3D instance
        """
        
        file = file_handle
        line = file.readline().split()                                          # Reads line and splits up data
        label = line[0]                                                         # Defines label from input data
        mass = float(line[1])                                                   
        position = np.array(line[2:5]).astype(np.float)                         # Creates position array from input data, converts to float
        velocity = np.array(line[5:8]).astype(np.float)                         # Creates velocity array from input data, converts to float
        
        return Planet3D( label, mass, position, velocity )                      # Returns particle instance



    def __str__(self):
        
        """
        XYZ-compliant string. The format is
        
        <label>    <x>  <y>  <z>
        """
        
        x = self.position[0]                                                    # Defines each position component 
        y = self.position[1]
        z = self.position[2]
    
        return str(self.label) + " " + str(x) + " " + str(y) + " " + str(z)     # Returns label and position components in string format

 

    def kinetic_e(self):
        
        """
        Kinetic energy of a Planet3D instance
        
        :return ke: float, 1/2 m v**2
        """
        
        return (1/2) * self.mass * (np.linalg.norm(self.velocity))**2           # Calculates kinetic energy using magnitude of velocity
        
    
    
    def momentum(self):
        
        """
        Linear momentum of a Particle3D instance
        
        :return p: (3) float np.array, m*v
        """
        
        return self.mass * self.velocity                                        # Calculates linear momentum 
                                                            
    
    
    def update_pos_1st(self, dt):
        
        """
        1st order position update
 
        :param dt: timestep
        :return position after one timestep: (3) float np.array, position + dt * velocity
        """
       
        self.position = self.position + dt * self.velocity                      # Calculates first order position update for a single planet
                               
        
        
    def update_pos_2nd(self, dt, force):
        
        """
        2nd order position update

        :param dt: timestep
        :param force: [3] float array, the total force acting on the planet
        :return position after two timesteps: (3) float np.array, position + dt * velocity + (dt^2) * (force/2*mass)
        """
       
        self.position = self.position + dt * self.velocity + (dt)**2 * (force/(2*self.mass))     # Calculates second order position update for a single planet



    def update_vel(self, dt, force):
        
        """
        Velocity update

        :param dt: timestep
        :param force: [3] float array, the total force acting on the planet
        :return updated velocity: (3) float np.array, velocity + dt * (force/mass)
        """
        
        self.velocity = self.velocity + dt * (force/self.mass)                  # Calculates velocity update for a planet given a force and timestep
        
                                                   

    @staticmethod
    def sys_kinetic(p3d_list):
        
        """
        Total kinetic energy of the whole system

        :param p3d_list: list in which each item is a P3D instance
        :return sys_ke: (sum 1/2 m_i v_i^2)
        """
        
        ke_list = []                                                            # List for the kinetic energy of each planet
        
        for i in range(0,len(p3d_list)):                                        # Calculates the kinetic energy of each planet and adds to ke list
            
            particle_ke = p3d_list[i].kinetic_e()
            #print(particle_ke)
            ke_list.append(particle_ke)
            
        #print(ke_list) 
        return sum(ke_list)                                                     # Returns summed list of planet kinetic energies, system kinetic energy
        
    
    
    @staticmethod
    def com_velocity(p3d_list):
        
        """
        Computes the CoM velocity of a list of PLanets

        :param p3d_list: list in which each item is a P3D instance
        :return total_mass: float, the total mass of the system, (sum m_i)
        :return com_vel: float, centre-of-mass velocity, (sum m_i*v_i)/(sum m_i) 
        """
        
        momentum_list =[]                                                       # Momentum and total mass lists for planets
        total_mass_list =[]
        
        for i in range(len(p3d_list)):
            
            particle_linear_momentum = p3d_list[i].momentum()                   # Calculates momentum for each planet
            
            momentum_list.append(particle_linear_momentum)                      # Adds individual momentums to momentum list
            total_mass_list.append(p3d_list[i].mass)                            # Adds individual planet masses to total_mass_list
            
            
        return sum(momentum_list)/sum(total_mass_list)                          # Returns CoM velocity
    
    
    
    @staticmethod
    def update_pos_2nd_list(p3d_list, dt, force): 
        
        """
        2nd order position update for a list of objects
        
        :param p3d_list: list of planet objects
        :param dt: timestep
        :param force: [3] float array, the total force acting on the planet 
        :return updates position after two timesteps for a list of objects 
        """
        
        updated_pos = []

        for i in range(len(p3d_list)):
            
            new_pos = p3d_list[i].update_pos_2nd(dt, force[i])                  # Updates every planet to new 2nd order positions in list
            updated_pos.append(new_pos)
            
        return updated_pos                                                      # Returns planets with updated positions
         
    
    
    @staticmethod
    def update_vel_list(p3d_list, dt, force): 
        
        """
        Velocity update for a list of objects

        :param p3d_list: list of planet objects
        :param dt: timestep
        :param force: [3] float array, the total force acting on the planet
        :returns updated velocity: (3) float np.array, velocity + dt * (force/mass)
        """
        
        resultant_vel = []

        for i in range(len(p3d_list)):                                        
            
            new_vel = p3d_list[i].update_vel(dt, force[i])                      # Updates every planet velocity in list
            resultant_vel.append(new_vel)
            
        return resultant_vel                                                    # Returns planets with updated velocities
    
    
    
    @staticmethod
    def Velocity_Correction(p3d_list, CoM_Velocity):
        
        """
        Velocity correction for a list of planets

        :param p3d_list: list of planet objects
        :param CoM_Velocity: Centre of Mass velocity vector
        :return Velocity Correction: list of planets with corrected velocites 
        """
        
        for i in range(len(p3d_list)):
            
            p3d_list[i].velocity = p3d_list[i].velocity - CoM_Velocity          # Subtracts CoM velocity from all planets velocities 
        
        return p3d_list

        