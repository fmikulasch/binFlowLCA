function [l2norml,energy,l1probr]= errorcurve(erroranalysis,output_dir,errorfile_dir,plot_flag)
% Energy curve
if nargin<4
    plot_flag=0;
end
if erroranalysis
    side={'Left','Right'}
    %%
    for i=1:8
        name = strjoin([errorfile_dir,side(mod(i,2)+1),'ErrorL2NormEnergyProbeAxis',num2str(floor((i-1)/2)),'.txt'],'')
        l2normA=importdata(name);
        if i==1
            l2norm = l2normA.data;
        else
            l2norm = l2norm + l2normA.data;
        end
    end
    %l2normright=textread([errorfile_dir,'RightErrorL2NormEnergyProbe.txt'],'%f');
    l1prob=importdata([errorfile_dir,'V1L1NormEnergyProbe.txt']);
    l1prob=l1prob.data;
    energy=importdata([errorfile_dir,'V1EnergyProbe.txt']);
    energy=energy.data(:,3);
    ind=rem([1:length(l2norm)-1],80)==0;
    ind(1)=0;
    q=-1;
    l2norml=l2norm(find(ind==1)+q);
    l1probr=l1prob(find(ind==1)+q);
    energy=energy(find(ind==1)+q);
    if plot_flag==1
        if length(energy)>5*10^6
            error('The data size is too big to make the plots')
        else
            figure('name','L2Norm')
            plot(l2norm);
            figure('name','L1norm')
            plot(l1prob)
            figure('name','Energy')
            plot(energy)
        end
        
    else
        figure ('name','Energy combined');
        hold on
        plot(l2norml,'r');
        plot(energy,'g');
        plot(l1probr,'b');
        legend ('L2 Norm', 'Energy', 'L1 Norm')
        saveas(gcf, [output_dir,'AllEnergy'] ,'png');
    end
end
end

%errorcurve(true,'../analysis/','../output/batchsweep_00/',0)

