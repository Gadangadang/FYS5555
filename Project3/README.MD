# Project 3
### Deployment of unsupervised learning for new physics at ATLAS with OPENDATA

This repository contains code and results for analyzing 2 lepton data from ATLAS <br>
Open Data. This project aims to understand how to use auto encoders for anomaly detection <br>
in high energy physics.<br>
The folders are structured ín the following way: <br>
<ol>
<li>Codebase contains notebooks with the machine learning and data handling</li>
<li>Article contains the report written for this project </li>
<li>figures contains the results from the different runs. Split into relevant folder for different tries </li>
<li>tf_models contains trained networks </li>
</ol>
<br>
The preselection was done via ROOT, and stored on a server, but the steps to recreate the data are written <br>
in the report, and the 2 lepton dataset can be found at the [ATLAS Open Data website](https://atlas.cern/resources/opendata). <br>
The machine learning is done using tensorflow 2 architecture, and the all the necessary tuning and training <br> 
parameters are shown in the report. 
