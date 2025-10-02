function [] = Two_attarctors_model()
%% Plot for Fig4

clear;close all

%% create vector filed for the model
Ystart = 1.5; % starting Y value
Ythr   = 15;  % goal

% Define attractor positions
attractor1 = [Ystart/5, Ystart/5];
attractor2 = [Ythr/5, Ythr/5];

% Define grid
[x, y] = meshgrid(linspace(0.1, 8, 40), linspace(0.1, 8, 40));
 
% Define relative strengths (depths) of the attractors
strength1 = 1;  % x vector
strength2 = 0.5;  % y vector
bias_y = 0;

% Define vector field with different strengths
vx = -strength1 * (x - attractor1(1)) .* exp(-((x - attractor1(1)).^2 + (y - attractor1(2)).^2)) ...
     -strength1 * (x - attractor2(1)) .* exp(-((x - attractor2(1)).^2 + (y - attractor2(2)).^2));
vy = -strength2 * (y - attractor1(2)) .* exp(-((x - attractor1(1)).^2 + (y - attractor1(2)).^2)) ...
     -strength2 * (y - attractor2(2)+bias_y) .* exp(-((x - attractor2(1)).^2 + (y - attractor2(2)).^2));

% increase pull as it goes higher  
modX = linspace(0,2,40)';
modX = modX.^1;
mod  = repmat(modX,1,40);
vx = vx.*mod;

% Normalize vector field
v_norm = sqrt(vx.^2 + vy.^2);
vx = vx ./ (v_norm + 1e-6);
vy = vy ./ (v_norm + 1e-6);

% scale it
X=x*5;Y=y*5;U=vx*30;V=vy*10;
x = X(1,:);y=X(1,:);

scale = 1;titles='';
f1 = plotVectorField(X,Y,U,V,Ystart,Ythr,scale,titles);

% fig_file_name = fullfile('plots',['VectorFild']);
% print(gcf,'-dtiff' ,fig_file_name)
% savefig(gcf,fig_file_name)



%% simulation of traj along the vector field

dt= 0.001;numStep=3000;

%% simulate trajectory

rots = 1:2; % angle of cue input

for rot=rots
    
     if rot==2
            amp = 220; % amp of the biggest input
            ty{2}=6; % step of inputs
     elseif rot==1
            amp = 150; % amp of the biggest input
            ty{1}=4;   % step of inputs
     end
        
    for i=1:ty{rot}
        initial_state = [1.5 1.5];

        step = 22.5; % step in amplitide between trials
        amp_end = amp-step*(i-1);
        
        sang = atan(1/1);
        if rot==2
            default_ang = sang - pi*0.1;
            ang = default_ang - pi*0.01*(i-1);
        else
            ang = sang;
        end
        
        pertureb_onset  = 0.5; % cue onset
        perturb_dur     = 0.2; % cue dur
        common_range    = perturb_dur; % s, input ramp down after this range (if shorter the perturb dur)
               
        inputs = [amp_end ang common_range pertureb_onset perturb_dur numStep];
        
        [xs{rot,i},ys{rot,i},t2]   = vf_simulation(x,y,U,V,initial_state,dt,inputs);
    end
end


%% plot results
for rot=rots
    
    numCond = 12;
    cs = turbo(numCond);
    cs(ty{rot},:) = [165 42 42]/255; % no lcik trial
    
    
    %% 2D plot
    f1 = plotVectorField(X,Y,U,V,Ystart,Ythr,scale,titles);
    hold on
    for i=1:ty{rot}
        plot(xs{rot,i},ys{rot,i},'o','linewidth',1,'color',cs(i,:),'markersize',2)
    end
    plot(Ystart,Ystart,'ko','markersize',12.5,'markerfacecolor','w')
    plot(Ythr,Ythr,'ko','markersize',12.5,'markerfacecolor','k')
    
    
    axis_xy = linspace(1.5,28.5,5);
    set(gca,'xtick',axis_xy,'ytick',axis_xy,'xticklabel',{'0',[],'1',[],'2'},...
        'yticklabel',{'0',[],'1',[],'2'})
%     fig_file_name = fullfile('plots',['VectorFild_traj_',num2str(rot)]);
%     print(gcf,'-dtiff' ,fig_file_name)
%     savefig(gcf,fig_file_name)


    %% projection
    fig = figure;set(gcf,'Color','w','Position',[44 184 300 600])
    tAxis = [1:numel(ys{rot,i})]*dt-0.5;
    subplot(2,1,1);hold on
    for i=1:ty{rot}
        plot(tAxis,ys{rot,i},'Color',cs(i,:),'linewidth',1.5)       
    end
    xlim([-0.2 1]);xline(0,'k:')
    xlabel('Time (s)');ylabel('Ramping mode');set(gca,'fontsize',16,'tickdir','out')
    set(gca,'ytick',axis_xy,'yticklabel',{'0',[],'1',[],'2'})
    
    subplot(2,1,2);hold on
    for i=1:ty{rot}
        plot(tAxis,xs{rot,i},'Color',cs(i,:),'linewidth',1.5)
    end
    xlim([-0.2 1]);xline(0,'k:')
    xlabel('Time (s)');ylabel('Cue mode');set(gca,'fontsize',16,'tickdir','out')
    set(gca,'ytick',axis_xy,'yticklabel',{'0',[],'1',[],'2'})
    
%     fig_file_name = fullfile('plots',['Ramp_mode',num2str(rot)]);
%     print(gcf,'-dtiff' ,fig_file_name)
%     savefig(gcf,fig_file_name)
    
    %% angle/amplitude at 200ms
    tID = find(tAxis>=0.2,1);
    LT  = nan(ty{rot},1); 
    for i=1:ty{rot}
        x_c{i} = (xs{rot,i}(tID)-Ystart)/(Ythr-Ystart);
        y_c{i} = (ys{rot,i}(tID)-Ystart)/(Ythr-Ystart);
        LTid   = find(ys{rot,i}>Ythr*0.99,1);
        if ~isnan(LTid)
            LT(i)  = tAxis(LTid);
        end
        
    end
    
    
    X_range = ty{rot};
    colors = turbo(X_range);
    ang = nan(X_range,1);amp= nan(X_range,1);
    figure;set(gcf,'Color','w','Position',[122 260 350 700]);
    for c=1:X_range
        subplot(3,1,1);hold on
        plot(x_c{c},y_c{c},'o','color',colors(c,:))

        u = [x_c{c} y_c{c}];
        v= [1 1]; % attarctor
       
        ang(c)   = atan2(det([u;v]),dot(u,v));      
        amp(c,:) = sqrt(x_c{c}.^2 + y_c{c}.^2);
    end

    subplot(3,1,2);hold on
    meanY = ang;

    plot(LT,meanY,'ko')
    xlabel('Lick time')
    ylabel('Angle (radian)')
    set(gca,'tickdir','out','box','off');
    yline(0,'k:')
    yline(ang(end),'k:')

    subplot(3,1,3);hold on
    meanY = amp;
    yline(amp(end),'k:')

    plot(LT,meanY,'ko')
    xlabel('Lick time')
    ylabel('Amplitude (a.u,)')
    set(gca,'tickdir','out','box','off');
    
%     fig_file_name = fullfile('plots',['TwoDPlotsAngQuat',num2str(rot)]);
%     print(gcf,'-dtiff' ,fig_file_name)
%     savefig(gcf,fig_file_name)
    
end


end


function [fig] = plotVectorField(X,Y,U,V,Ystart,Ythr,scale,titles)
    % plot vector filed
    fig = figure;set(gcf,'Color','w','Position',[70 520 300 300]);hold on
    quiver(X,Y,U,V,scale,'Color',[0.5 0.5 0.5],'linewidth',1.5)
    xlim([0 30])
    ylim([1 16])
    yline(Ythr,'k:')
    yline(Ystart,'k:')
    
    xlabel('Cue mode');ylabel('Ramping mode');set(gca,'fontsize',16,'tickdir','out')
%     set(gca,'ytick',1.5:((15-1.5)/3):15,'yTickLabel',{'0','','','1'})
%     set(gca,'xtick',0.5:((17.5-0.5)/3):17.5,'xTickLabel',{'0','','','1'})
    title(titles)
end




function [x,y,t] = vf_simulation(X,Y,Utmp,Vtmp,initial_state,dt,inputs)

    % inputs = [amp ang amp_end common_range pertureb_onset perturb_dur Ythr numStep];
   

        
    perturbOnsetBin = inputs(4)/dt;
    perturbDurBin   = inputs(5)/dt;
    numStep         = inputs(6);
    
    

    x = nan(numStep,1);x(1) = initial_state(1);
    y = nan(numStep,1);y(1) = initial_state(2);
    
    for t=2:numStep

     
        if iscell(Utmp)
            U = Utmp{t};V = Vtmp{t};
        else
            U = Utmp;V = Vtmp;
        end
        
        xind   = x(t-1);
        xpost  = ceil(xind);
        xpre   = xpost-1;
        if xpre==0;xpre=1;end
        
        yind  = y(t-1);
        ypost  = ceil(yind);
        ypre   = ypost-1;
        if ypre==0;ypre=1;end
        
        U1 = (xpost-xind)*U(ypre,xpre) + (xind-xpre)*U(ypre,xpost);
        U2 = (xpost-xind)*U(ypost,xpre) + (xind-xpre)*U(ypost,xpost);         
        xSpeed = (ypost-yind)*U1 + (yind-ypre)*U2;
        
        V1 = (xpost-xind)*V(ypre,xpre) + (xind-xpre)*V(ypre,xpost);
        V2 = (xpost-xind)*V(ypost,xpre) + (xind-xpre)*V(ypost,xpost);         
        ySpeed = (ypost-yind)*V1 + (yind-ypre)*V2;
        
        if isempty(perturbOnsetBin)
        
            x(t) = x(t-1)+xSpeed*dt;
            y(t) = y(t-1)+ySpeed*dt;

        else

            if t>=perturbOnsetBin && t<=perturbOnsetBin+perturbDurBin
               
                amp  = inputs(1);
                ang  = inputs(2);
                common_range = inputs(3);
                
                
                tdiff = t-(perturbOnsetBin+perturbDurBin*common_range);
                
                if tdiff>0
                    % ramp down of input
                    mod = (tdiff/(perturbDurBin*(1-common_range)));       
                    amp2 = amp*(1-mod);
                else
                    amp2 = amp;
                end
                

                stimStrength = [amp2*cos(ang) amp2*sin(ang)];
                
                x(t) = x(t-1)+(xSpeed+stimStrength(1))*dt;
                y(t) = y(t-1)+(ySpeed+stimStrength(2))*dt;

             
            else
                x(t) = x(t-1)+xSpeed*dt;
                y(t) = y(t-1)+ySpeed*dt;
            end

            if y(t)<1;y(t)=1;end
            if x(t)<1;x(t)=1;end


        end
        
%         if y(t)>=Ythr || x(t)<1;break;end
    end


end

