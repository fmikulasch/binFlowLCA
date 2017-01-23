%output_dir='/run/user/1000/gvfs/sftp:host=vr-lab.am28.uni-tuebingen.de,user=neuralnetlab/home/neuralnetlab/workspace/Projects/BinFlowLCA/testOutput/'
output_dir='/run/user/1000/gvfs/sftp:host=vr-lab.am28.uni-tuebingen.de,user=neuralnetlab/media/Data/testOutput/'
dir_sel_mon=[134,26,73,37]+1;
dir_sel_bin=[158,109,167,91]+1;
mot_sen=[63,207]+1;
fw_trans=[130,18,146,2]+1;
bw_trans=[65,33,97,1]+1;
cw_rot=[132,36,164,4]+1;
ccw_rot=[72,24,88,8]+1;
spec_cells={dir_sel_mon,dir_sel_bin,mot_sen,fw_trans,bw_trans,cw_rot,ccw_rot};
colors={'blue','blue','blue','green','green','green','green'};

[data,header]=readpvpfile(strcat(output_dir,'V1.pvp'));
n=8
groups = zeros(2^n,n);
for i = 1:2^n
    groups(i,:)=dec2bin(i-1,n);
end
groups=groups-48; %???

%groups(mot_sen,:)
v1pack=[];
for i = 2: 9
    v1container=zeros(header.nx*header.ny*header.nf,1);
    v1container(data{i}.values(:,1)+1)=data{i}.values(:,2);
    v1pack(:,i-1)=squeeze(sum(sum(reshape(v1container,header.nf,header.nx,header.ny),3),2));
end

additionalGroups = [4 6 5 1 2 8 3 7 5 1 7 6 2 4 8 3 8 7 2 3 5 1 4 6 2 6 8 3 1 5 4 7]
for i = 1 : length(additionalGroups) 
    v1container=zeros(header.nx*header.ny*header.nf,1);
    v1container(data{i+9}.values(:,1)+1)=data{i + 9}.values(:,2);
    v1pack(:,additionalGroups(i))=v1pack(:,additionalGroups(i))+squeeze(sum(sum(reshape(v1container,header.nf,header.nx,header.ny),3),2));
end

neurgroups=zeros(1,length(v1container));
for i = 1: length(v1container) % find group of neuron
    neuract=v1pack(i,:);
    co=zeros(1,2^n);
    for g = 1: 2^n
        s=0;
        ma=max(neuract);
        for j = 1: length(neuract)
           if neuract(j) == 0 && groups(g,j) == 0
               s = s + 1;
           else
                   s = s + neuract(j) * groups(g,j);
           end
        end
        co(g)=s;
    end
    f = find(co==max(co));
    neurgroups(i)=f(1);
end
test=cat(2,v1pack,groups(neurgroups,:));
test=cat(2,test,neurgroups');
groupocc = zeros(1,2^n);
for i = 1:length(groupocc)
    groupocc(i) = sum(neurgroups==i);
end
groupocc(1)=0; %drop neurons that do not fire

% logic operations


plotflag=1;
if plotflag~=0

    activityplot=figure;
    subplot(12,1,2:11);
    bar(groupocc);
    xlim([0 2^n]);
    xlabel('Index of group');
    ylabel('Number of neurons in group');
    set(gca,'XAxisLocation','top');
    axis tight;
    hold on;
    subplot(12,1,12);
    barwidth=1;
    bar(1:length(groupocc),ones(size(groupocc)),'facecolor','red','barwidth',barwidth);
    hold on;
    xlim([0 2^n]);
    for i = 1:length(spec_cells)
        v=cell2mat(spec_cells(i));
        bar(v,ones(size(groupocc(v))),cell2mat(colors(i)),'barwidth',barwidth/min(diff(sort(v))));
        hold on;
    end

    activityplot=figure;
    [sortout,ind]=sort(groupocc,'descend');
    ind(1)
    unsorted = 1:length(sortout);
    newInd(ind) = unsorted;
    subplot(12,1,2:11);
    bar(sortout);
    xlim([0 2^n]);
    xlabel('Index of group');
    ylabel('Number of neurons in group');
    set(gca,'XAxisLocation','top');
    axis tight;
    hold on;
    subplot(12,1,12);
    barwidth=1;
    bar(1:length(sortout),ones(size(sortout)),'facecolor','red','barwidth',barwidth);
    hold on;
    xlim([0 2^n]);
    for i = 1:length(spec_cells)
        v=newInd(cell2mat(spec_cells(i)));
        bar(v,ones(size(sortout(v))),cell2mat(colors(i)),'barwidth',barwidth/min(diff(sort(v))));
        hold on;
    end

    %saveas(activityplot,'kernel activity.png')

end