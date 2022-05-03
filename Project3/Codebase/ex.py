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
    
    columns["met"].append(np.mean(a))
   
    columns["XSection"].append(np.mean(a))  
    ### Lep information
    columns["lep_n"].append
    columns["tot_lep_invariant_mass"].append(np.mean(a))
    columns["mean_lep_pt"].append(np.mean(a))
    columns["mean_lep_E"].append(np.mean(a))
    columns["mean_lep_ptcone30"].append(np.mean(a))
    columns["mean_lep_etcone20"].append(np.mean(a))
    columns["mean_lep_eta"].append(np.mean(a))
    columns["mean_lep_phi"].append(np.mean(a))

    ### Jet information

    columns["jet_n"].append(np.mean(a))
    columns["mean_jet_pt"].append(np.mean(a))
    columns["mean_jet_E"].append(np.mean(a))
    columns["mean_jet_eta"].append(np.mean(a))
    columns["mean_jet_phi"].append(np.mean(a))
    columns["mean_jet_pt_syst"].append(np.mean(a))

    ### Photon information

    columns["photon_n"].append(np.mean(a))
    columns["mean_photon_pt"].append(np.mean(a))
    columns["mean_photon_E"].append(np.mean(a))
    columns["mean_photon_ptcone30"].append(np.mean(a))
    columns["mean_photon_etcone20"].append(np.mean(a))
    columns["mean_photon_eta"].append(np.mean(a))
    columns["mean_photon_phi"].append(np.mean(a))
    columns["mean_photon_pt_syst"].append(np.mean(a))

    ### LargeRjet information

    columns["largeRjet_n"].append(np.mean(a))
    columns["tot_largeRjet_m"].append(np.mean(a))
    columns["mean_largeRjet_pt"].append(np.mean(a))
    columns["mean_largeRjet_E"].append(np.mean(a))
    columns["mean_largeRjet_eta"].append(np.mean(a))
    columns["mean_largeRjet_phi"].append(np.mean(a))
    columns["mean_largeRjet_pt_syst"].append(np.mean(a))


    ### Tau information 

    columns["tau_n"].append(np.mean(a))
    columns["mean_tau_pt"].append(np.mean(a))
    columns["mean_tau_E"].append(np.mean(a))
    columns["mean_tau_eta"].append(np.mean(a))
    columns["mean_tau_phi"].append(np.mean(a))
    columns["mean_tau_pt_syst"].append(np.mean(a))


    ### Systematic uncertainty for gauranteed placeholders
    columns["mean_lep_pt_syst"].append(np.mean(a))
    columns["met_et_syst"].append(np.mean(a))