import numpy as np 
import pandas as pd

a = np.array([1,2,3,4])

a1 = 1
a2 = 2

n = int(1e5)
columns = {"met":[], "XSection":[], 
           "lep_n":[],"tot_lep_invariant_mass":[], "mean_lep_pt":[], 
           "mean_lep_E":[], "mean_lep_ptcone30":[], "mean_lep_etcone20":[],
           "mean_lep_eta":[], "mean_lep_phi":[], "jet_n":[], 
           "mean_jet_pt":[], "mean_jet_E":[], "mean_jet_eta":[], 
           "mean_jet_phi":[], "photon_n":[], "mean_photon_pt":[], 
           "mean_photon_E":[], "mean_photon_ptcone30":[], 
           "mean_photon_etcone20":[],"mean_photon_eta":[], 
           "mean_photon_phi":[], "largeRjet_n":[], "tot_largeRjet_m":[],
           "mean_largeRjet_pt":[], "mean_largeRjet_E":[], 
           "mean_largeRjet_eta":[], "mean_largeRjet_phi":[],
           "tau_n":[], "mean_tau_pt":[], "mean_tau_E":[], 
           "mean_tau_eta":[], "mean_tau_phi":[],
           "mean_lep_pt_syst":[], "met_et_syst":[], 
           "mean_jet_pt_syst":[], "mean_photon_pt_syst":[], 
           "mean_largeRjet_pt_syst":[], "mean_tau_pt_syst":[]
          }


for i in range(n):
    
    columns["met"].append((a[a1] + a[a2])/2)
   
    columns["XSection"].append((a[a1] + a[a2])/2)  
    ### Lep information
    columns["lep_n"].append
    columns["tot_lep_invariant_mass"].append((a[a1] + a[a2])/2)
    columns["mean_lep_pt"].append((a[a1] + a[a2])/2)
    columns["mean_lep_E"].append((a[a1] + a[a2])/2)
    columns["mean_lep_ptcone30"].append((a[a1] + a[a2])/2)
    columns["mean_lep_etcone20"].append((a[a1] + a[a2])/2)
    columns["mean_lep_eta"].append((a[a1] + a[a2])/2)
    columns["mean_lep_phi"].append((a[a1] + a[a2])/2)

    ### Jet information

    columns["jet_n"].append((a[a1] + a[a2])/2)
    columns["mean_jet_pt"].append((a[a1] + a[a2])/2)
    columns["mean_jet_E"].append((a[a1] + a[a2])/2)
    columns["mean_jet_eta"].append((a[a1] + a[a2])/2)
    columns["mean_jet_phi"].append((a[a1] + a[a2])/2)
    columns["mean_jet_pt_syst"].append((a[a1] + a[a2])/2)

    ### Photon information

    columns["photon_n"].append((a[a1] + a[a2])/2)
    columns["mean_photon_pt"].append((a[a1] + a[a2])/2)
    columns["mean_photon_E"].append((a[a1] + a[a2])/2)
    columns["mean_photon_ptcone30"].append((a[a1] + a[a2])/2)
    columns["mean_photon_etcone20"].append((a[a1] + a[a2])/2)
    columns["mean_photon_eta"].append((a[a1] + a[a2])/2)
    columns["mean_photon_phi"].append((a[a1] + a[a2])/2)
    columns["mean_photon_pt_syst"].append((a[a1] + a[a2])/2)

    ### LargeRjet information

    columns["largeRjet_n"].append((a[a1] + a[a2])/2)
    columns["tot_largeRjet_m"].append((a[a1] + a[a2])/2)
    columns["mean_largeRjet_pt"].append((a[a1] + a[a2])/2)
    columns["mean_largeRjet_E"].append((a[a1] + a[a2])/2)
    columns["mean_largeRjet_eta"].append((a[a1] + a[a2])/2)
    columns["mean_largeRjet_phi"].append((a[a1] + a[a2])/2)
    columns["mean_largeRjet_pt_syst"].append((a[a1] + a[a2])/2)


    ### Tau information 

    columns["tau_n"].append((a[a1] + a[a2])/2)
    columns["mean_tau_pt"].append((a[a1] + a[a2])/2)
    columns["mean_tau_E"].append((a[a1] + a[a2])/2)
    columns["mean_tau_eta"].append((a[a1] + a[a2])/2)
    columns["mean_tau_phi"].append((a[a1] + a[a2])/2)
    columns["mean_tau_pt_syst"].append((a[a1] + a[a2])/2)


    ### Systematic uncertainty for gauranteed placeholders
    columns["mean_lep_pt_syst"].append((a[a1] + a[a2])/2)
    columns["met_et_syst"].append((a[a1] + a[a2])/2)