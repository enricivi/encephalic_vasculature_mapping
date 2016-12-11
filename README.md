# EncephalicVasculatureMapping
This work aims to explore a new approach to model the encephalic vasculature using the formalism of graphs that naturally fit the structure of blood vessels.

To obtain the graph follow the procedure below:

0) If necessary make_train_set.py (folder: augment-train-set) performs an augmenting operation over the train-set  
1) Use train_vascular.py to train the neural network  
2) Use predict.py to segment the images  
3) It is possibile to use overlap_img.py (folder: pre-process-denoising) to overlap the predicted image (It is possibile that    this operation reduces the noise)  
4) Use prepare_tif_stack.py to filter and binarize the predicted images  
5) Use the Fiji's (Fiji is just ImageJ) plug-in to create the graph that models the predicted images. To run the plug-in move    in the directory containing the launcher and the run (command for linux)  
   $ ./ImageJ-linux64 --ij2 --headless --run path/to/script 'img_file="path/to/image",ouput="path/to/output/folder"'  
   Fiji: https://fiji.sc/#  
   Headless mode: http://imagej.net/Scripting_Headless  
6) Use cluster_meanshift.py (folder: post-process-denoising) to clean the graph from noises (with the bandwith's value of 15      the result is very good)  

NB use -h to see the script's helps
