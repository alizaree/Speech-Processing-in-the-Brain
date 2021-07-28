function [model] = mTRFmodelLoader(resp,stim, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs)
    % this function takes resp(neural responses as a cell of trials where each element of the
    % cell, each trial, is a matrix an it is assumed that the rows
    % correspond to observations and the columns to variables.
    % same thing goes for stim which is the corresponding stimulus to the
    % resp.
    % dir_name is the address to save the trained model.
    % load_prev_dec (if 1) indicates whether to load the existing model in
    % dir_name for prediction or to retrain a model and overwrite the model
    % in dir_name. In case no such model exists in the directory or load_prev_dec is 0, a model
    % is trained and saved and used for prediction.
    % lambdas are a list of values for regularization and controlling overfitting default value is: 10.^(-6:1:6)
 
%     Pass in 1 for map to fit a forward model, or -1 to fit a backward
%     model. STIM and RESP are matrices or cell arrays containing
%     corresponding trials of continuous training data. Fs is a scalar
%     specifying the sample rate in Hertz, and Tmin (default 0) and Tmax (default 650) are scalars
%     specifying the minimum and maximum time lags in milliseconds. For
%     backward models, mTRFtrain automatically reverses the time lags.
 
% output is the trained model.

% author of the code: Ali Zare, Mar 23, 2021. 
    

    if ~exist('load_prev_dec','var') || isempty(load_prev_dec)
    load_prev_dec=1;
    end
    
    if ~exist('lambdas','var') || isempty(lambdas)
    lambdas=10.^(-6:1:6);
    end
    
    if ~exist('tmin','var') || isempty(tmin)
        tmin=-0;
    end
    
    if ~exist('tmax','var') || isempty(tmax)
        tmax=650;
    end
    
    if ~exist('map','var') || isempty(map)
        map=1; % forward
    end
    
    if ~exist('Fs','var') || isempty(Fs)
        Fs=100; % forward
    end
    
    RUNS=1:length(resp);
    if load_prev_dec && isfile(dir_name)
        load(dir_name)
    else

        [stats,t]=mTRFcrossval(stim,resp,Fs,map,tmin,tmax,lambdas,'fast',1);    %Use crossval function to find optimal lambda
        [~,L]=max(mean(mean(stats.r(:,:,:),3),1));    %Optimal lambda found from the highest r value
        % p = stats.p;
        % r(p>0.001)=NaN;
        % % p(p>0.001)=NaN;
        % % p = squeeze(nanmean(p,1));
        % r = squeeze(nanmean(r,1));
        % [~,LMDA]=max(mean(r,2));   %Optimal lambda found from the highest r value
        % % elecs = find(p(LMDA,:)<0.01);
        % % [~,LMDA]=max(mean(mean(r(:,:,:),3),1));   %Optimal lambda found from the highest r value
    %     LLA = lambdas(L)
    %     LLU = lambdas(LU);
        % clear model r
        L=lambdas(L);
        disp(['found lambda: ', num2str(L), '. starting training']);
        for runs=RUNS
            disp(['trial', ' ', num2str(runs), '/', num2str(length(RUNS))]);
            dd=setdiff(RUNS,runs);
            [model(runs)]=mTRFtrain(vertcat(stim{dd}),vertcat(resp{dd}),Fs,map,tmin,tmax,L);    % Train the model. Hold out one trial for the test set and train on the rest

        end
        save(dir_name, 'model');
    end
end

