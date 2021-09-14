# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:31:21 2020

@author: Benjamin
"""


import numpy as np
#import h5py
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numba import jit
from scipy.ndimage.interpolation import rotate
import math as m



#potentials = h5py.File('set_of_potentials_small.h5', 'w')  # Create file object

'''Set empty dictionary for appending data'''
int_energy_data = []
int_energy_per_atom_data = []
sum_ind_atom_energies = []
difference_energy = []
potential_data=[]
'''Set the number of grid points in the x,y,z direction''' 
Nx=50              # Number of points in x-direction
Ny=50             # Number of points in y-direction
Nz=50             # Number of points in z-direction
'''Hyper parameter for psuedo potential'''
gamma = 0.36         # Angstrom
'''Set the grid spacing - smaller provides a finer grid'''
grid_space = 0.155   # Angstrom
'''Physical length if the grid in Angstrom'''
Lx = Nx*grid_space  # Angstrom
Ly = Ny*grid_space  # Angstrom
Lz = Nz*grid_space  # Angstrom
'''To offset the 3D grid 0,0,0 to the center of cube'''
offset = Lx/2       # The offset for "COM" coordinates.

#%%
'''Function to extract the data from the text files'''
def get_data(file_name):
    
    fin=open(file_name)
    
    first = fin.readline().split()              # Reads first line. How many atoms there are.
    no_atoms = int(first[0])                    # Gets the interger value for number of atoms
    
    frame = np.zeros((no_atoms,4))              # Create frame to store the atomic number and coordinates
    
    info = fin.readline().split()               # Reads second line containing the physical information.
    internal_energy_0K = float(info[12])        # Extracts the internal energy at 0K.
    
    id_number = int(info[1])
    
    temp_dict = []                              # creates a dictionary to store the atomic species and coords in string format
    ind_energy = 0.
    for i in range(no_atoms):
 
        temp_dict = [] 
        temp_dict = fin.readline().split()
        
        if temp_dict[0]== 'H':                  # Here we check the atomic species and assign the atomic number.
            frame[i,0] = 1
            ind_energy += -0.5
        elif temp_dict[0]== 'C':
            frame[i,0] = 6
            ind_energy += -37.8450
        elif temp_dict[0]== 'N':
            frame[i,0] = 7
            ind_energy += -54.5892
        elif temp_dict[0]== 'O':
            frame[i,0] = 8
            ind_energy += -75.0673
            
        frame[i,1] = np.single(temp_dict[1])      # Insert x coordinate
        frame[i,2] = np.single(temp_dict[2])         # Insert y coordinate
        frame[i,3] = np.single(temp_dict[3])       # Insert z coordinate
    
    
    diff_energy = internal_energy_0K - ind_energy

    return no_atoms,internal_energy_0K,frame,id_number,ind_energy,diff_energy

'''
Function to obtain the pseudo Guassian potential
Nx,Ny,Nz - Number of grid points
no_atoms - Number of atoms in molecule
atomic_info - co-ordinates of the atom[l]
grid_space - The length of the grid/matrix element
offset_x,y,z - Off set the x,y,z axis to the center of the matrix
id_number - id number of the molecule
'''
@jit(nopython=True)
def gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number):
    
    V_pot = np.zeros((Nx,Ny,Nz)) # Initalise the grid for Gaussian Potential
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                V_term_space = np.zeros(no_atoms) # create a space for the summation term of length according to number of atoms.

                for l in range(no_atoms):
                    
                    V_term_space[l] = atomic_info[l,0] *\
                        np.exp(   (-1./(2*(gamma)**2))* (  ( ((i*grid_space)-offset+offset_x)-atomic_info[l,1] )**2 +  \
                                        (((j*grid_space)-offset+offset_y)-atomic_info[l,2])**2 + \
                                            (((k*grid_space)-offset+offset_z)-atomic_info[l,3])**2   )   )
                            
                    
                    
                V_pot[i,j,k] = np.single(np.sum(V_term_space))      #sum the terms in the function and make it single precision.
                
                # np.savetxt('Gaussian_potential'+str(id_number)'.txt',V_pot)
    return V_pot

def Rx(phi):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(phi),-m.sin(phi)],
                   [ 0, m.sin(phi), m.cos(phi)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(psi):
  return np.matrix([[ m.cos(psi), -m.sin(psi), 0 ],
                   [ m.sin(psi), m.cos(psi) , 0 ],
                   [ 0           , 0            , 1 ]])


def rotate_matrix(atomic_info,phi,theta,psi):
    
    R = Rx(phi)*Ry(theta)*Rz(psi)
    
    for l in range(len(atomic_info[:,0])):
        atomic_info[l,1:4] = atomic_info[l,1:4]*R
    return atomic_info
    


#%%

for ii in range(1,10):
    try:             #The try: "code" ---> except:pass will continue the loop if error is found.
        print(ii)
        got_data = get_data('/home/mcnaughton/ML_Work/Atoms_and_energy_dataset/dsgdb9nsd_00000'+str(ii)+'.xyz')
        no_atoms = got_data[0]
        internal_energy =  got_data[1]
        atomic_info = got_data[2]
        id_number = got_data[3]
        ind_energy = got_data[4]
        diff_energy = got_data[5]
        
        offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
        offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
        offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
        
        sizex = (np.amax(atomic_info[:,1]) - np.amin(atomic_info[:,1]))
        sizey = (np.amax(atomic_info[:,2]) - np.amin(atomic_info[:,2]))
        sizez = (np.amax(atomic_info[:,3]) - np.amin(atomic_info[:,3]))
        
        size = np.sqrt(sizex**2. + sizey**2. + sizez**2.)
        
#        if no_atoms <= 12:
        if sizex <= 5.0 and sizey <= 5.0 and sizez <= 5.0:

            
            my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            potential_data.append(my_pot)
            del(my_pot)
            del(offset_x)
            del(offset_y)
            del(offset_z)
            int_energy_data.append(internal_energy)
            int_energy_per_atom_data.append(internal_energy/no_atoms)
            sum_ind_atom_energies.append(ind_energy)
            difference_energy.append(diff_energy) 
            
            for i in range(0):
            
            
                phi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                theta = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                psi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                
                atomic_info= rotate_matrix(atomic_info,phi,theta,psi)
                
                offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
                offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
                offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
                
                my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            
                potential_data.append(my_pot)
                int_energy_data.append(internal_energy)
                int_energy_per_atom_data.append(internal_energy/no_atoms)
                sum_ind_atom_energies.append(ind_energy)
                difference_energy.append(diff_energy)
                del(my_pot)
                del(offset_x)
                del(offset_y)
                del(offset_z)
      
    except:
        pass
    
#%%    
for ii in range(10,100):
    try:             #The try: "code" ---> except:pass will continue the loop if error is found.
        print(ii)
        got_data = get_data('/home/mcnaughton/ML_Work/Atoms_and_energy_dataset/dsgdb9nsd_0000'+str(ii)+'.xyz')
        no_atoms = got_data[0]
        internal_energy =  got_data[1]
        atomic_info = got_data[2]
        id_number = got_data[3]
        ind_energy = got_data[4]
        diff_energy = got_data[5]
        
        offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
        offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
        offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
        
        sizex = (np.amax(atomic_info[:,1]) - np.amin(atomic_info[:,1]))
        sizey = (np.amax(atomic_info[:,2]) - np.amin(atomic_info[:,2]))
        sizez = (np.amax(atomic_info[:,3]) - np.amin(atomic_info[:,3]))
        
        size = np.sqrt(sizex**2. + sizey**2. + sizez**2.)
        
#        if no_atoms <= 12:
        if sizex <= 5.0 and sizey <= 5.0 and sizez <= 5.0:

            
            my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            potential_data.append(my_pot)
            del(my_pot)
            del(offset_x)
            del(offset_y)
            del(offset_z)
            int_energy_data.append(internal_energy)
            int_energy_per_atom_data.append(internal_energy/no_atoms)
            sum_ind_atom_energies.append(ind_energy)
            difference_energy.append(diff_energy) 
            
            for i in range(0):
            
            
                phi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                theta = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                psi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                
                atomic_info= rotate_matrix(atomic_info,phi,theta,psi)
                
                offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
                offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
                offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
                
                my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            
                potential_data.append(my_pot)
                int_energy_data.append(internal_energy)
                int_energy_per_atom_data.append(internal_energy/no_atoms)
                sum_ind_atom_energies.append(ind_energy)
                difference_energy.append(diff_energy)
                del(my_pot)
                del(offset_x)
                del(offset_y)
                del(offset_z)
      
    except:
        pass
#%%    
for ii in range(100,1000):
    try:             #The try: "code" ---> except:pass will continue the loop if error is found.
        print(ii)
        got_data = get_data('/home/mcnaughton/ML_Work/Atoms_and_energy_dataset/dsgdb9nsd_000'+str(ii)+'.xyz')
        no_atoms = got_data[0]
        internal_energy =  got_data[1]
        atomic_info = got_data[2]
        id_number = got_data[3]
        ind_energy = got_data[4]
        diff_energy = got_data[5]
        
        offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
        offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
        offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
        
        sizex = (np.amax(atomic_info[:,1]) - np.amin(atomic_info[:,1]))
        sizey = (np.amax(atomic_info[:,2]) - np.amin(atomic_info[:,2]))
        sizez = (np.amax(atomic_info[:,3]) - np.amin(atomic_info[:,3]))
        
        size = np.sqrt(sizex**2. + sizey**2. + sizez**2.)
        
#        if no_atoms <= 12:
        if sizex <= 5.0 and sizey <= 5.0 and sizez <= 5.0:

            
            my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            potential_data.append(my_pot)
            del(my_pot)
            del(offset_x)
            del(offset_y)
            del(offset_z)
            int_energy_data.append(internal_energy)
            int_energy_per_atom_data.append(internal_energy/no_atoms)
            sum_ind_atom_energies.append(ind_energy)
            difference_energy.append(diff_energy) 
            
            for i in range(0):
            
            
                phi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                theta = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                psi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                
                atomic_info= rotate_matrix(atomic_info,phi,theta,psi)
                
                offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
                offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
                offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
                
                my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            
                potential_data.append(my_pot)
                int_energy_data.append(internal_energy)
                int_energy_per_atom_data.append(internal_energy/no_atoms)
                sum_ind_atom_energies.append(ind_energy)
                difference_energy.append(diff_energy)
                del(my_pot)
                del(offset_x)
                del(offset_y)
                del(offset_z)
      
    except:
        pass
    
for ii in range(1000,10000):
    try:             #The try: "code" ---> except:pass will continue the loop if error is found.
        print(ii)
        got_data = get_data('/home/mcnaughton/ML_Work/Atoms_and_energy_dataset/dsgdb9nsd_00'+str(ii)+'.xyz')
        no_atoms = got_data[0]
        internal_energy =  got_data[1]
        atomic_info = got_data[2]
        id_number = got_data[3]
        ind_energy = got_data[4]
        diff_energy = got_data[5]
        
        offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
        offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
        offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
        
        sizex = (np.amax(atomic_info[:,1]) - np.amin(atomic_info[:,1]))
        sizey = (np.amax(atomic_info[:,2]) - np.amin(atomic_info[:,2]))
        sizez = (np.amax(atomic_info[:,3]) - np.amin(atomic_info[:,3]))
        
        size = np.sqrt(sizex**2. + sizey**2. + sizez**2.)
        
#        if no_atoms <= 12:
        if sizex <= 5.0 and sizey <= 5.0 and sizez <= 5.0:

            
            my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            potential_data.append(my_pot)
            del(my_pot)
            del(offset_x)
            del(offset_y)
            del(offset_z)
            int_energy_data.append(internal_energy)
            int_energy_per_atom_data.append(internal_energy/no_atoms)
            sum_ind_atom_energies.append(ind_energy)
            difference_energy.append(diff_energy) 
            
            for i in range(0):
            
            
                phi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                theta = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                psi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                
                atomic_info= rotate_matrix(atomic_info,phi,theta,psi)
                
                offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
                offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
                offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
                
                my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            
                potential_data.append(my_pot)
                int_energy_data.append(internal_energy)
                int_energy_per_atom_data.append(internal_energy/no_atoms)
                sum_ind_atom_energies.append(ind_energy)
                difference_energy.append(diff_energy)
                del(my_pot)
                del(offset_x)
                del(offset_y)
                del(offset_z)
      
    except:
        pass
    
for ii in range(10000,100000):
    try:             #The try: "code" ---> except:pass will continue the loop if error is found.
        print(ii)
        got_data = get_data('/home/mcnaughton/ML_Work/Atoms_and_energy_dataset/dsgdb9nsd_0'+str(ii)+'.xyz')
        no_atoms = got_data[0]
        internal_energy =  got_data[1]
        atomic_info = got_data[2]
        id_number = got_data[3]
        ind_energy = got_data[4]
        diff_energy = got_data[5]
        
        offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
        offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
        offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
        
        sizex = (np.amax(atomic_info[:,1]) - np.amin(atomic_info[:,1]))
        sizey = (np.amax(atomic_info[:,2]) - np.amin(atomic_info[:,2]))
        sizez = (np.amax(atomic_info[:,3]) - np.amin(atomic_info[:,3]))
        
        size = np.sqrt(sizex**2. + sizey**2. + sizez**2.)
        
#        if no_atoms <= 12:
        if sizex <= 5.0 and sizey <= 5.0 and sizez <= 5.0:

            
            my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            potential_data.append(my_pot)
            del(my_pot)
            del(offset_x)
            del(offset_y)
            del(offset_z)
            int_energy_data.append(internal_energy)
            int_energy_per_atom_data.append(internal_energy/no_atoms)
            sum_ind_atom_energies.append(ind_energy)
            difference_energy.append(diff_energy) 
            
            for i in range(0):
            
            
                phi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                theta = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                psi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                
                atomic_info= rotate_matrix(atomic_info,phi,theta,psi)
                
                offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
                offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
                offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
                
                my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            
                potential_data.append(my_pot)
                int_energy_data.append(internal_energy)
                int_energy_per_atom_data.append(internal_energy/no_atoms)
                sum_ind_atom_energies.append(ind_energy)
                difference_energy.append(diff_energy)
                del(my_pot)
                del(offset_x)
                del(offset_y)
                del(offset_z)
      
    except:
        pass
    
# #%%
# for ii in range(100000,133886):
#     try:             #The try: "code" ---> except:pass will continue the loop if error is found.
#         print(ii)
#         got_data = get_data('/home/mcnaughton/ML_Work/Atoms_and_energy_dataset/dsgdb9nsd_'+str(ii)+'.xyz')
#         no_atoms = got_data[0]
#         internal_energy =  got_data[1]
#         atomic_info = got_data[2]
#         id_number = got_data[3]
#         ind_energy = got_data[4]
#         diff_energy = got_data[5]
        
#         offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
#         offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
#         offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
        
#         sizex = (np.amax(atomic_info[:,1]) - np.amin(atomic_info[:,1]))
#         sizey = (np.amax(atomic_info[:,2]) - np.amin(atomic_info[:,2]))
#         sizez = (np.amax(atomic_info[:,3]) - np.amin(atomic_info[:,3]))
        
#         size = np.sqrt(sizex**2. + sizey**2. + sizez**2.)
        
# #        if no_atoms <= 12:
#         if sizex <= 5.0 and sizey <= 5.0 and sizez <= 5.0:

            
#             my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
#             potential_data.append(my_pot)
#             del(my_pot)
#             del(offset_x)
#             del(offset_y)
#             del(offset_z)
#             int_energy_data.append(internal_energy)
#             int_energy_per_atom_data.append(internal_energy/no_atoms)
#             sum_ind_atom_energies.append(ind_energy)
#             difference_energy.append(diff_energy) 
            
#             for i in range(0):
            
            
#                 phi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
#                 theta = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
#                 psi = (m.pi*np.random.randint(11))/(np.random.randint(10)+1)
                
#                 atomic_info= rotate_matrix(atomic_info,phi,theta,psi)
                
#                 offset_x = np.sum(atomic_info[:,1])/len(atomic_info[:,1])
#                 offset_y = np.sum(atomic_info[:,2])/len(atomic_info[:,2])
#                 offset_z = np.sum(atomic_info[:,3])/len(atomic_info[:,3])
                
#                 my_pot = gaussian_pot(Nx,Ny,Nz,no_atoms,atomic_info,gamma,grid_space,offset_x,offset_y,offset_z,id_number)
            
#                 potential_data.append(my_pot)
#                 int_energy_data.append(internal_energy)
#                 int_energy_per_atom_data.append(internal_energy/no_atoms)
#                 sum_ind_atom_energies.append(ind_energy)
#                 difference_energy.append(diff_energy)
#                 del(my_pot)
#                 del(offset_x)
#                 del(offset_y)
#                 del(offset_z)
      
#     except:
#         pass
    


np.savetxt('internal_energies_medium_atoms.txt',int_energy_data,fmt='%1.7f')
np.savetxt('internal_energies_per_atom_medium_atoms.txt',int_energy_per_atom_data,fmt='%1.7f')
np.savetxt('individual_atoms_energy_summed.txt',sum_ind_atom_energies,fmt='%1.7f')
np.savetxt('difference_gs_summed_energies.txt',difference_energy,fmt='%1.7f')
np.save('medium_atoms_potential.npy',potential_data)


#%%

################ Plotting Code ########################
# levs = np.linspace(0.,50.,200, dtype = float) #This produces the contour levels.

# def make_ticklabels_invisible(fig):
#     for i, ax in enumerate(fig.axes):
#         #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)

# for ii in range(len(potential_data)):
#     pot_to_plot = potential_data[ii]
#     print('energy is',int_energy_data[ii])
#     for iii in range(Nx):
#     ##### Image Plot ############
#         fig = plt.figure(figsize=(11,11))
#         #fig.suptitle('H='+str(i*0.1)+'Hc2')  
#         plt.imshow(pot_to_plot[iii,:,:,])                      # Change slice indices to check the potential [x,y,z]
#         #make_ticklabels_invisible(fig)
#         plt.show()
    
#    #### Contour Plot ##########
#    fig = plt.figure(figsize=(11,11))
#    #fig.suptitle('H='+str(i*0.1)+'Hc2')  
#    plt.contour(my_pot[10,:,:],levels=levs)         # Change slice indices to check the potential [x,y,z]  
#    #make_ticklabels_invisible(fig)
#    plt.show()
