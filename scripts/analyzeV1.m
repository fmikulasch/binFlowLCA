%%  analyzeV1.m - An analysis tool for first-layer PV implementations
%%  -Wesley Chavez
%%
%% -----Hints-----
%% Copy this file to your PV output directory (outputPath in params file) and edit lines 26-30 accordingly.
%% 
%% Write all layers and connections at the end of a display period (initialWriteTime = displayPeriod*n or displayPeriod*n - 1 and writeStep = displayPeriod*k, n and k are integers > 0)
%% Sync the write times of Input and Recon layers for comparison (writeStep and initialWriteTime in params file).
%% Sync the write times of Input and Error layers for more useful RMS values (more useful than just standard deviation of Error values).
%% Sync the write times of Error and V1 layers for Error vs V1sparsity graph.
%% You know what, just sync all your write times.
%%
%% If you want to run this script while your PetaVision implementation is still running, don't change the order of readpvpfile commands.
%%
%% -----ToDo-----
%% Add functionality to read from Checkpoint pvp files (no write time synchronization needed).
%% Convolve 2nd layer (V2,S2,etc.) weights with 1st layer (V1,S1) weights for visualization in image space.
%% Figure out how to plot/save everything without user input (Octave asks "-- less -- (f)orward, (b)ack, (q)uit" after plotting). 


% Counts number of figures plotted, initialize to zero (Doesn't include input and recon imwrites)
numFigures = 0; 

addpath('~/workspace/OpenPV/mlab/util'); 

outputPath = '../testOutput/'
side = {'Left' , 'Right'};

for s = 1:1
   for q = 1:1
	fflush(1);
	inputpvp = strjoin([side(s) , 'ImageAxis', num2str(q) ,'.pvp'], '')
	errpvp = strjoin([side(s) , 'ErrorAxis', num2str(q) ,'.pvp'], '')
	V1pvp = 'V1.pvp'
	reconpvp = strjoin([side(s) , 'ReconAxis', num2str(q) ,'.pvp'], '')

	%How many of the most recent inputs/recons in the pvp files you want to write
	numImagesToWrite = 5;



	%%%% Input    Only last numImagesToWrite frames are written 
	inputdata = readpvpfile([outputPath,inputpvp],10);

	% Save these for error computation
	for i = 1:size(inputdata,1)
	   t_input(i) = inputdata{i}.time;
	   inputstd(i) = std(inputdata{i}.values(:));
	end

	% Normalize and write input images
	for i = size(inputdata,1)-numImagesToWrite+1:size(inputdata,1)
	   t = inputdata{i}.time;
	   p = inputdata{i}.values;
	   disp (['Ganglion size: ', num2str(size(p))]);
	   p = p-min(p(:));
	   p = p*255/max(p(:));
	   p = permute(p,[2 1 3]);
	   p = uint8(p);
	   outFile = ['Ganglion_' sprintf('%.08d',t) '.png']
	   imwrite(p,outFile);
	end
	clear inputdata;



	%%%% Recon    Only last numImagesToWrite frames are read and written
	fid = fopen([outputPath,reconpvp],'r');
	reconheader = readpvpheader(fid);
	fclose(fid);
	if (reconheader.nbands < numImagesToWrite)
	   display('Recon pvp was only written to %d times, but numImagesToWrite is specified as %d\n', reconheader.nbands, numImagesToWrite);
	end
	recondata = readpvpfile([outputPath,reconpvp],10, reconheader.nbands, reconheader.nbands-numImagesToWrite+1);

	% Normalize and write recon images
	for i = 1:size(recondata,1)
	   t = recondata{i}.time;
	   p = recondata{i}.values;
	   disp (['Recon size: ', num2str(size(p))]);
	   p = p-min(p(:));
	   p = p*255/max(p(:));
	   p = permute(p,[2 1 3]);
	   p = uint8(p);
	   outFile = ['Recon_' sprintf('%.08d',t) '.png']
	   imwrite(p,outFile);
	end
	clear recondata;



	%%%% Error
	%% If write-times for input layer and error were synced, plot RMS error.  Else, plot std of error values. 
	errdata = readpvpfile([outputPath,errpvp],10);


	for i = 1:size(t_input,2)  % If PetaVision implementation is still running, errdata might contain more frames, even if synced with input, since errpvp is read after inputpvp. 
	   if (errdata{i}.time == t_input(i))
	      syncedtimes = 1;
	   else
	      syncedtimes = 0;
	      break;
	   end
	end

	if (syncedtimes)
	   for i = 1:size(t_input,2)
	      t_err(i) = errdata{i}.time;
	      err(i) = std(errdata{i}.values(:))/inputstd(i);
	   end
	   numFigures++;
	   h_err = figure(numFigures);
	   plot(t_err,err);
	   outFile = ['RMS_Error_' sprintf('%.08d',t_err(length(t_err))) '.png']
	   print(h_err,outFile);
	else
	   for i = 1:size(errdata,1)
	      t_err(i) = errdata{i}.time;
	      err(i) = std(errdata{i}.values(:));
	   end
	   numFigures++;
	   h_err = figure(numFigures);
	   plot(t_err,err);
	   outFile = ['Std_Error_' sprintf('%.08d',t_err(length(t_err))) '.png']
	   print(h_err,outFile);
	end
	clear errdata;



	%%%% V1 Sparsity and activity per feature
	[V1data V1header] = readpvpfile([outputPath,V1pvp],10);
	V1numneurons = V1header.nx*V1header.ny*V1header.nf;
	V1meanfeaturevals = [];
	for i = 1:size(V1data,1)
	   t_V1(i) = V1data{i}.time;
	   V1sparsity(i) = size(V1data{i}.values,1)/(V1numneurons);
	   %if (i == size(V1data,1))
	   if (t_V1(i) == V1data{size(V1data,1)}.time)
	      V1_yxf = zeros(1,V1numneurons);
	      V1_yxf(V1data{i}.values(:,1)+1) = V1data{i}.values(:,2);
	      V1_yxf = reshape(V1_yxf,[V1header.nf V1header.nx V1header.ny]);
	      V1_yxf = permute(V1_yxf,[3 2 1]);  % Reshaped to actual size of V1 layer
	      if isempty(V1meanfeaturevals)
		 V1meanfeaturevals = V1_yxf;
		 V1meanfeaturevals = V1meanfeaturevals(:)';
	      else
		 V1meanfeaturevals = V1meanfeaturevals + V1_yxf(:)';
	      end
	      t_V1_sortedweights = t_V1(i);
	   end   
	end
	numFigures++;
	h_V1 = figure(numFigures);
	plot(t_V1,V1sparsity);
	outFile = ['V1_Sparsity_' sprintf('%.08d',t_V1(length(t_V1))) '.png']
	print(h_V1,outFile);

	numFigures++;
	h_V1featvals = figure(numFigures);
	bar(V1meanfeaturevals);
	outFile = ['MeanFeatureValues_' sprintf('%.08d',t_V1(length(t_V1))) '.png']
	print(h_V1featvals,outFile);
	clear V1data;



	%%%% Error vs Sparse    Print this graph if V1 and Error write times are synced. (blue = first write time, red = last write time)
	for i = 1:size(t_err,2)  % If PetaVision implementation is still running, V1data might contain more frames, even if synced with input, since V1pvp is read after errpvp.
	   if (t_V1(i) == t_err(i))
	      syncedtimes = 1;
	   else
	      syncedtimes = 0;
	      break;
	   end
	end

	if (syncedtimes)
	   numFigures++;
	   h_ErrorvsSparse = figure(numFigures);
	   c=linspace(0,1,length(err));
	   scatter(V1sparsity(1:length(err)),err,[],c);
	   xlabel('Sparsity');
	   ylabel('Error');
	   outFile = ['ErrorVsSparse_' sprintf('%.08d',t_V1(length(t_V1))) '.png']
	   print(h_ErrorvsSparse,outFile);
	end



	%%%% Weights: Only last weights frame is analyzed. Each weightspatch is normalized in respect to other Weights.
	weightsheaders = cell(2,4);
	weightsfiledatas = cell(2,4);
	weightsdatas = cell(2,4);
	for o = 1:2
           for p = 1:4
		weightspvp = strjoin(['V1To', side(o) , 'ErrorAxis', num2str(p-1) ,'.pvp'], '');
		weightspvp
		fflush(1);
		fid = fopen([outputPath,weightspvp],'r');
		weightsheaders{o,p} = readpvpheader(fid);
		fclose(fid);
		weightsfiledatas{o,p} = dir([outputPath,weightspvp]);
		weightsframesize = weightsheaders{1,1}.recordsize*weightsheaders{1,1}.numrecords+weightsheaders{1,1}.headersize;
		weightsnumframes = weightsfiledatas{1,1}.bytes/weightsframesize;
		weightsdatas{o,p} = readpvpfile([outputPath,weightspvp],10,weightsnumframes,weightsnumframes);
	   end
	end
	
	weightsnumpatches = size(weightsdatas{1,1}{size(weightsdatas{1,1},1)}.values{1})(4)
	num_patches_rows = floor(sqrt(weightsnumpatches));
	num_patches_cols = ceil(weightsnumpatches / num_patches_rows);
	minmaxweights = cell(weightsnumpatches,2);
	minmaxweights(:,1) = 1000000;
	minmaxweights(:,2) = 0;

	for o = 1:2
           for p = 1:4
		for i = 1:weightsnumpatches
		   tmp = weightsdatas{o,p}{size(weightsdatas{o,p},1)}.values{1}(:,:,:,i);
		   minmaxweights{i,1} = min(min(tmp(:)), minmaxweights{i,1});
		   minmaxweights{i,2} = max(max(tmp(:)), minmaxweights{i,2});
		end
	   end
	end

	for o = 1:2
           for p = 1:4
		for i = 1:weightsnumpatches
		   weightspatch{o,p,i} = weightsdatas{o,p}{size(weightsdatas{o,p},1)}.values{1}(:,:,:,i);
		   weightspatch{o,p,i} = weightspatch{o,p,i}-minmaxweights{i,1};
		   weightspatch{o,p,i} = weightspatch{o,p,i}*(255/minmaxweights{i,2});
		   weightspatch{o,p,i} = uint8(permute(weightspatch{o,p,i},[2 1 3]));
		end
	   end
	end
	
	weight_patch_array = [];
	[dontcare sortedindex] = sort(V1meanfeaturevals);
	sortedindex = fliplr(sortedindex);
	for o = 1:2
           for p = 1:4
		for i = 1:weightsnumpatches
		   patch = weightspatch{o,p,sortedindex(i)};
		   if isempty(weight_patch_array)
		   	weight_patch_array = ...
		   	zeros(num_patches_rows*size(patch,1), num_patches_cols*size(patch,2), size(patch,3));
		   end
		   col_ndx = 1 + mod(i-1, num_patches_cols);
		   row_ndx = 1 + floor((i-1) / num_patches_cols);
		   weight_patch_array(((row_ndx-1)*size(patch,1)+1):row_ndx*size(patch,1), ...
		                      ((col_ndx-1)*size(patch,2)+1):col_ndx*size(patch,2),:) = ...
		   patch;
		end
		imwrite(uint8(weight_patch_array), ['weights', num2str(4*(o-1)+p), '.png'], 'png');
		weight_patch_array = ...
		zeros(num_patches_rows*size(patch,1), num_patches_cols*size(patch,2), size(patch,3));
	   end
	end
   end
end
