%% forward model from resp to BERT features individual and combined with fixed acoustic:

%% 1- forward model based on clean for subjects with clean based on all old features. (zscoring) 
% ****** New Features added, trained on acoustic, acst+phn, acst+phn+phon,
% acst+phn+phon+sem
clc
clear all
%% path for freesurf
global globalFsDir;
globalFsDir='/Applications/freesurfer/7.1.1/subjects/';
%% loading the files
jjj=1 ;
total_C=0;
total_C=total_C+1;
sbj={'LIJ109', 'LIJ110', 'LIJ112', 'LIJ113', 'LIJ114', 'LIJ120', 'LIJ123'};
sbj=sbj{jjj};
if jjj==6
    sbjN=[];
else
sbjN=sbj;
end

clean_pre_name='/out_clean_';
I_pre_name= '/out_HG_'; 
if jjj==6
    avg='avg_';
else
    avg=[];
end
zscoring=2;% WARNING: there are two of it. 1 is for zscore 2 is for dividing by norm. Keep two if pca was applied before
contDvD=1; % concatinate triales and divide them into equal trials (1) or not (0)
compresSpg=1; % 1 for 16 frequency 0 for 100 and smth
if compresSpg==0
    AllChn='AllChn';
else
    AllChn=[];
end
hannLengthB=7; % default is 7 ****
hannN=['hann',num2str(hannLengthB)];
dir_name_clean=['../Data/',sbj,clean_pre_name,avg,sbj,'_',AllChn,'AllandorGPT2FNWsentL_2LYR_Norm_pca_zsc',num2str(zscoring),'hann',num2str(hannLengthB),'.mat'];
dir_name_I=['../Data/',sbj,I_pre_name,avg,sbj,'_',AllChn,'AllandorGPT2FNWsentL_2LYR_Norm_pca_zsc',num2str(zscoring),'hann',num2str(hannLengthB),'.mat'];
gptname='GPT2andSpgFNWsentL_2LYR_nrmPC_WndCont';
zscoring=2; %  **************
    %{
    clean_add=['../Data/', sbj, clean_pre_name,sbj,'.mat'];
    clean=load(clean_add);

    I_add=['../Data/', sbj,I_pre_name,sbj,'.mat'];
    load(I_add);
    %}
    
% clean=load(dir_name_clean);
load(dir_name_I);
out=outNew; 
if jjj==6
elecs120=out(1).elecs;
end
%elecs{total_C}=out(1).elecs;

if contDvD==1
    % concatinate triales and divide them into equal trials
    outNew=ConcAndDivide(outNew); % secc is empty then 20 trials if -1 then 1 trial
%     clean.outNew=ConcAndDivide(clean.outNew);
    % make double numbers single

    Clean_name='_clean_ContDvD_';
    I_name='_NoisyAll_ContDvD_';
    All_name='_All_ContDvD_';
else
    list1=[];
    list2=[];
    for i=1:length(outNew)
        if outNew(i).duration<20

            list1=[list1 i];
        end
%         if clean.outNew(i).duration<20
% 
%             list2=[list2 i];
%         end
    end
    outNew(list1)=[];
%     clean.outNew(list2)=[];
    Clean_name='_cleanD_';
    I_name='_NoisyAll_';
    All_name='_All_';
end

sameStim=0 ; % DO NOT CHANGE THIS.
if sameStim==1
    NsameStim='sameStm';
else
    NsameStim=[];
end

outNew=makeSingle(outNew);
%clean.outNew=makeSingle(clean.outNew);
    
    

  
%% make the stim and resp file for training or predicting 

load_prev_dec=1;
include_stopWrds=1; 
if include_stopWrds==0
    stpWrdN='_stpWrd0';
else
    stpWrdN='_stpWrd1';
end
addNoise=0; % add noise to the moments when we don't have stim
if addNoise==1
    nameNcond='NoiseOnly';
else
    nameNcond='ZeroOnly';
end
nshfl=20;
Manuall_Fix_Weight_edge=0;
if Manuall_Fix_Weight_edge==1
    edgeFixM='WEdgeF';
else
    edgeFixM=[];
end
Fs=100;
noiseNs={'bar', 'city', 'jet', 'clean'};
post_adapt=500*10^-3;% time for noise adaptation.in sec
if post_adapt==500*10^-3
    NpostAdapt=[];
else
    NpostAdapt=num2str(post_adapt);
end
% for noise=1:4 %noises={'bar', 'city', 'jet', 'clean'};
%     noise_name=['n_',noiseNs{noise}];
% 
%     
% [startT,stopT]=FindtimeSwitch(outNew.label,[],1);
% times=[];
% for inst=1:length(startT)
%     times=[times; ((startT(inst)+post_adapt*Fs ):stopT(inst))'];
% end
% end

tmin=-0;
tmax=700;
lambdas= 10.^(1:6);%10.^(-6:1:6);
%     lambdas = [1 10 100];
nLYR=size(outNew(1).GPT2_act,3);
map=1; % 1 forward, -1 backward
if map==1
    model_name='forward_model';
    mn='fw';
    stim_PCA=1;
    resp_PCA=0;
else
    model_name='backward_model';
    mn='bw';
    resp_PCA=0;
    stim_PCA=0;
end
resp = [];
respN= [];
Window=[-1,0, 1 , 2, 3, 4,5,10,-2];
nWindow=length(Window);
%% stimState represents if it's just spg, GPT2 or all

% nSteps=floor( (length(times)-window_dur*Fs)/(stepL*Fs) )+1;
% tIn= (cnt1-1)*stepL*Fs+1:(cnt1-1)*stepL*Fs+window_dur*Fs;
% tOut= setxor(1:length(times), tIn);
wnd=9;
if wnd==2
    nameWnd=[];
else
    nameWnd=['_FxdWnd',num2str(Window(wnd))];
end
nameWnd=['_FxdWnd',num2str(Window(wnd))];
            
for cnt1 = 1:length(outNew)
    tmpStimall = [];
    % ttt=outNew(cnt1).label==4; % only using the times when the voice is clean. 
    tmpResp = outNew(cnt1).resp;
%     for wnd=1:nWindow
        for layer=1:nLYR
            tmpStim = squeeze(outNew(cnt1).GPT2_act(:,wnd, layer,:));
            stimTmp{layer}{cnt1} = (tmpStim./sqrt(size(tmpStim,2)));%/norm(tmpStimAall(:));
        end
%     end
    resp{cnt1}=(tmpResp);

end
%{
    % making the clean stim for times when there is no noise present
for cnt1 = 1:length(outNew)
    tmpStimall = [];
    % ttt=outNew(cnt1).label==4; % only using the times when the voice is clean. 
    [startT,stopT]=FindtimeSwitch(outNew(cnt1).label==4 | outNew(cnt1).label==0,1);
    times=[];
    for inst=1:length(startT)
        times=[times; ((startT(inst)+post_adapt*Fs ):stopT(inst))'];
    end
    othertimes=setxor(1:size(outNew(cnt1).resp,1),times);
    tmpResp = outNew(cnt1).resp;
    for wnd=1:nWindow
        for layer=1:nLYR
            tmpStim = squeeze(outNew(cnt1).GPT2_act(:,wnd,layer,:));
            if addNoise==1
                stdNoise=std(tmpStim(othertimes,:),[],1);
                tmpStim(othertimes,:)=bsxfun(@times,randn(length(othertimes),size(tmpStim,2)),stdNoise);
            else
                tmpStim(othertimes,:)=0;
            end
            if include_stopWrds==0 % means don't include stop words
                tmpStim=tmpStim.*(outNew(cnt1).nonStopWrds>0);
            end
            stim{wnd}{layer}{cnt1} = (tmpStim./sqrt(size(tmpStim,2)));%/norm(tmpStimAall(:));
        end
    end
    resp{cnt1}=(tmpResp);

end

    % making the window trials resp and stim
for cnt1 = 1:length(outNew)
    tmpStimall = [];
    % ttt=outNew(cnt1).label==4; % only using the times when the voice is clean. 
    [startT,stopT]=FindtimeSwitch(outNew(cnt1).label~=4,1);
    times=[];
    for inst=1:length(startT)
        times=[times; ((startT(inst)+post_adapt*Fs ):stopT(inst))'];
    end
    othertimes=setxor(1:size(outNew(cnt1).resp,1),times);
    tmpRespN= outNew(cnt1).resp;
    for wnd=1:nWindow
        for layer=1:nLYR
            tmpStim = squeeze(outNew(cnt1).GPT2_act(:,wnd,layer,:));
            if addNoise==1
                stdNoise=std(tmpStim(othertimes,:),[],1);
                tmpStim(othertimes,:)=bsxfun(@times,randn(length(othertimes),size(tmpStim,2)),stdNoise);
            else
                tmpStim(othertimes,:)=0;
            end
            if include_stopWrds==0 % means don't include stop words
                tmpStim=tmpStim.*(outNew(cnt1).nonStopWrds>0);
            end
            stimN{wnd}{layer}{cnt1} = (tmpStim./sqrt(size(tmpStim,2)));%/norm(tmpStimAall(:));
        end
    end
    respN{cnt1}=(tmpRespN);

end
%}
% make resp shuffle
for shfll=1:nshfl
    order=randsample(length(outNew),length(outNew));
    for cnt1_sfl = 1:length(outNew)

        cnt1=order(cnt1_sfl);
        
        
        
        [startT,stopT]=FindtimeSwitch(outNew(cnt1).label==4 | outNew(cnt1).label==0,1);
        times=[];
        for inst=1:length(startT)
            times=[times; ((startT(inst)+post_adapt*Fs ):stopT(inst))'];
        end
        ttt=min(size(resp{cnt1},1), size(resp{cnt1_sfl},1));
        % ttt=outNew(cnt1).label==4; % only using the times when the voice is clean. 
        tmpResp = outNew(cnt1).resp(1:ttt,:);


        resp_sfl{cnt1_sfl,shfll}=(tmpResp);
        
        
        %{
        [startT,stopT]=FindtimeSwitch(outNew(cnt1).label~=4,1);
        times=[];
        for inst=1:length(startT)
            times=[times; ((startT(inst)+post_adapt*Fs ):stopT(inst))'];
        end
        ttt=min(size(resp{cnt1},1), size(resp{cnt1_sfl},1));
        % ttt=outNew(cnt1).label==4; % only using the times when the voice is clean. 

        tmpRespN= outNew(cnt1).resp(1:ttt,:);

        respN_sfl{cnt1_sfl,shfll}=(tmpRespN);
        %}
    end

end
RUNS=1:length(resp);



%% Apply PCA and combine
ncmp=size(stimTmp{1}{1},2);
%ncmp=10;
% normalize the stim and devide by sqrt(dim)
% for wnd=1:nWindow
    for layer=1:nLYR
        tmp=[];
        for trl=1:length(RUNS)
            tmp=[tmp; stimTmp{layer}{trl}(:,1:ncmp)];
        end

        if zscoring==2
            normtmp=vecnorm(tmp,2,1);
            for trl=1:length(RUNS)
                stimTmp{layer}{trl}=bsxfun(@rdivide,stimTmp{layer}{trl}(:,1:ncmp), normtmp);%stimTmp{layer}{trl}./ normtmp(1);%
                stimTmp{layer}{trl}=stimTmp{layer}{trl}./sqrt(size( stimTmp{layer}{trl},2));
            end
        elseif zscoring==1
            meantmp=mean(tmp,1);
            stdtmp=std(tmp,[],1);
            for trl=1:length(RUNS)
                stimTmp{layer}{trl}=bsxfun(@minus,stimTmp{layer}{trl}(:,1:ncmp), meantmp);
                stimTmp{layer}{trl}=bsxfun(@rdivide,stimTmp{layer}{trl}, stdtmp);%stimTmp{layer}{trl}./ stdtmp(1);%
                stimTmp{layer}{trl}=stimTmp{layer}{trl}./sqrt(size( stimTmp{layer}{trl},2));
            end
        end
    end 
% end
%{
ncmp=0;
if stim_PCA==1
    ncmp=20;
    stim_PCAs=[];
    pre_stim=stim;
    for layer=1:nLYR
        tempS=stimT{end}{layer};
        Stim{layer}=vertcat(tempS{:});
    end
    for layer=1:nLYR
        [~,PCA]=PCA_object(Stim{layer}, [],ncmp);
        stim_PCAs=[stim_PCAs, {PCA}];
    end
    for wnd=1:nWindow
        for layer=1:nLYR
            for trl=1:length(RUNS)
                stim{wnd}{layer}{trl}=PCA_object(stim{wnd}{layer}{trl}, stim_PCAs{layer});
            end
        end  
    end
     % normalize the stim and devide by sqrt(dim)
    for wnd=1:nWindow
        for layer=1:nLYR
            tmp=[];
            for trl=1:length(RUNS)
                tmp=[tmp; stim{wnd}{layer}{trl}];
            end

            if zscoring==2
                normtmp=vecnorm(tmp,2,1);
                for trl=1:length(RUNS)
                    stim{wnd}{layer}{trl}=stim{wnd}{layer}{trl}./ normtmp(1);%bsxfun(@rdivide,stim{layer}{trl}, normtmp);
                    stim{wnd}{layer}{trl}=stim{wnd}{layer}{trl}./sqrt(size( stim{wnd}{layer}{trl},2));
                end
            elseif zscoring==1
                meantmp=mean(tmp,1);
                stdtmp=std(tmp,[],1);
                for trl=1:length(RUNS)
                    stim{wnd}{layer}{trl}=bsxfun(@minus,stim{wnd}{layer}{trl}, meantmp);
                    stim{wnd}{layer}{trl}=bsxfun(@rdivide,stim{wnd}{layer}{trl}, stdtmp);
                    stim{wnd}{layer}{trl}=stim{wnd}{layer}{trl}./sqrt(size( stim{wnd}{layer}{trl},2));
                end
            end
        end 
    end

    
    for wnd=1:nWindow
        for layer=1:nLYR
            for trl=1:length(RUNS)
                stimN{wnd}{layer}{trl}=PCA_object(stimN{wnd}{layer}{trl}, stim_PCAs{layer});
            end
        end  
    end
     % normalize the stim and devide by sqrt(dim)
    for wnd=1:nWindow
        for layer=1:nLYR
            tmp=[];
            for trl=1:length(RUNS)
                tmp=[tmp; stimN{wnd}{layer}{trl}];
            end

            if zscoring==2
                normtmp=vecnorm(tmp,2,1);
                for trl=1:length(RUNS)
                    stimN{wnd}{layer}{trl}=stimN{wnd}{layer}{trl}./ normtmp(1);%bsxfun(@rdivide,stim{layer}{trl}, normtmp);
                    stimN{wnd}{layer}{trl}=stimN{wnd}{layer}{trl}./sqrt(size( stimN{wnd}{layer}{trl},2));
                end
            elseif zscoring==1
                meantmp=mean(tmp,1);
                stdtmp=std(tmp,[],1);
                for trl=1:length(RUNS)
                    stimN{wnd}{layer}{trl}=bsxfun(@minus,stimN{wnd}{layer}{trl}, meantmp);
                    stimN{wnd}{layer}{trl}=bsxfun(@rdivide,stimN{wnd}{layer}{trl}, stdtmp);
                    stimN{wnd}{layer}{trl}=stimN{wnd}{layer}{trl}./sqrt(size( stimN{wnd}{layer}{trl},2));
                end
            end
        end 
    end

end

% STIM{2}=stim;
% STIMN{2}=stimN;

 %}   
    
%% add spectrogram:


Allfields={'aud_sound', 'aud_D_sound', 'ph_on',...
'ph_nVec_tVec','ph_bigram_tVec',...
        'cohort_entr_tVec','cohort_surp_tVec','word_on','word_f_tVec',...
        'sem_den_tVec'};
Ling=struct('acoustic',1,'acoustic_D',2,'phnOn',3, 'phn',4,...
'phon',5,'leX',7,'wrdOn',8,'wrd_f',9, 'sem',10); 
lngN=fieldnames(Ling);


lng=length(lngN); 
allfields=Allfields(1:Ling.(lngN{1}));


indexEnv= find(strcmp(allfields, 'env')); 
indexPhIdx=find(strcmp(allfields, 'ph_idx_tVec')); 
indexDerv= find(strcmp(allfields,'aud_D_sound'));


for cnt1 = 1:length(RUNS)
        tmpStimall = []; 
        tmpStimNall=[];
%         [startT,stopT]=FindtimeSwitch(outNew(cnt1).label==4 |outNew(cnt1).label==0,1);
%         times=[]; 
%         for inst=1:length(startT)
%         times=[times; ((startT(inst)+post_adapt*Fs ):stopT(inst))'];
%         end
        for cnt2 = 1:length(allfields) 
            % no need since the relu is already applied %             
%         if ~isempty(indexDerv) && cnt2==indexDerv %                 
%             tmpStim =clean.outNew(cnt1).(allfields{cnt2}).*(clean.outNew(cnt1).(allfields{cnt2})>0);                
%             tmpStimN =outNew(cnt1).(allfields{cnt2}).*(outNew(cnt1).(allfields{cnt2})>0); % %
%         else %                 
%             tmpStim = clean.outNew(cnt1).(allfields{cnt2}); %
%             tmpStimN = outNew(cnt1).(allfields{cnt2}); %            
%         end

            tmpStim = outNew(cnt1).(allfields{cnt2});

            tmpStimall =cat(2,tmpStimall,tmpStim./sqrt(size(tmpStim,2)));

        end
%         for layer=  1:nLYR
%             stimAll{layer}{cnt1} =cat(2,stimTmp{layer}{cnt1},tmpStimall);%/norm(tmpStimAall(:));
% 
%         end

        stimA{cnt1} =(tmpStimall);%/norm(tmpStimAall(:))
        
        
        

end



tmp=[];
for trl=1:length(RUNS)
    tmp=[tmp; stimA{trl}];
end

if zscoring==2
    normtmp=vecnorm(tmp,2,1);
    for trl=1:length(RUNS)
        stimA{trl}=bsxfun(@rdivide,stimA{trl}, normtmp);%stimTmp{layer}{trl}./ normtmp(1);%
        stimA{trl}=stimA{trl}./sqrt(size( stimA{trl},2));
    end
elseif zscoring==1
    meantmp=mean(tmp,1);
    stdtmp=std(tmp,[],1);
    for trl=1:length(RUNS)
        stimA{trl}=bsxfun(@minus,stimA{trl}, meantmp);
        stimA{trl}=bsxfun(@rdivide,stimA{trl}, stdtmp);%stimA{trl}./ stdtmp(1);%
        stimA{trl}=stimA{trl}./sqrt(size( stimA{trl},2));
    end
end




for cnt1 = 1:length(RUNS)
        
        for layer=  1:nLYR
            stimAll{layer}{cnt1} =cat(2,stimTmp{layer}{cnt1},stimA{cnt1});%/norm(tmpStimAall(:));

        end
      
end

stimT{1}=stimA;
stimT{2}=stimTmp;
stimT{3}=stimAll;


% normalize the spec part ( applicabale only for this script because we
% denorm the spec by seperating times)
%{
all=cat(1,stimT{1}{:});
N=vecnorm(all);
for iRun=1:length(RUNS)
    tmp=stimT{1}{iRun};
    tmp=bsxfun(@rdivide,tmp./sqrt(size(all,2)),N);
    stimT{1}{iRun}=tmp;
end



for layer=1:nLYR
    all=cat(1,stimT{3}{layer}{:});
    N=vecnorm(all);
    for iRun=1:length(RUNS)
        tmp=stimT{3}{layer}{iRun};
        tmp(:,ncmp+1:end)=bsxfun(@rdivide,tmp(:,ncmp+1:end)./sqrt(size(all,2)-ncmp),N(ncmp+1:end));
        stimT{3}{layer}{iRun}=tmp;
    end
    
end

%}
    %% looking at the stims before training
    
%     figure()
%     plot(mean(stim{1}{1}(:,1:60), 2))
%     hold on
%     plot(mean(stim{1}{1}(:,61:76), 2))
%     plot(mean(stim{1}{1}(:,77:92), 2))
% 
%     plot(mean(stimN{1}{1}(:,1:60), 2))
%     plot(mean(stimN{1}{1}(:,61:76), 2))
%     plot(mean(stimN{1}{1}(:,77:92), 2))
% 
%     legend('GPT2C',' spgC', 'DspgC', 'GPT2N',' spgN', 'DspgN')
%     hold off

    %% saving loading model for clean
for stimState=1:3
%     
%     
     if stimState==2 || stimState==3
         
        
        for layer=1:nLYR
            dir_name=['../Data/',model_name,'/',mn,AllChn,All_name,sbj,gptname,'(',nameNcond,'_noAdapt',NpostAdapt,')',stpWrdN,'_stmState_',num2str(stimState),nameWnd,'layer_',num2str(layer),'_lags', num2str(tmin), num2str(tmax),'zsc',num2str(zscoring),hannN,'HG_SpeechRE_PCA',num2str(ncmp),'.mat'];
            model{stimState}{layer}=mTRFmodelLoader(resp,stimT{stimState}{layer}, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs);
        end
     else
         dir_name=['../Data/',model_name,'/',mn,AllChn,All_name,sbj,gptname,'(',nameNcond,'_noAdapt',NpostAdapt,')',stpWrdN,'_stmState_',num2str(stimState),nameWnd,'_lags', num2str(tmin), num2str(tmax),'zsc',num2str(zscoring),hannN,'HG_SpeechRE_PCA',num2str(ncmp),'.mat'];
         model{stimState}=mTRFmodelLoader(resp,stimT{stimState}, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs);

     end
        %{
            %% loading and saving the files for noise
        for layer=1:nLYR
%             if stimState==2
                dir_name=['../Data/',model_name,'/',mn,AllChn,I_name,sbj,gptname,'(',nameNcond,'_noAdapt',NpostAdapt,')',stpWrdN,'_CntxW_',num2str(Window(wnd)),'layer_',num2str(layer),'_lags', num2str(tmin), num2str(tmax),'zsc',num2str(zscoring),hannN,'HG_SpeechRE_PCA',num2str(ncmp),'.mat'];
%             else
%                 dir_name=['../Data/',model_name,'/',mn,AllChn,I_name,sbj,'AllandGPT2FN(NoiseOnly_noAdapt',NpostAdapt,')',NsameStim,'_state_',num2str(stimState),'layer_',num2str(layer),'_lags', num2str(tmin), num2str(tmax),'zsc',num2str(zscoring),hannN,'HG_SpeechRE_PCA',num2str(ncmp),'.mat'];
% 
%             end
            if sameStim==0
                 modelN{wnd}{layer}=mTRFmodelLoader(respN,stimN{wnd}{layer}, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs);
            elseif sameStim==1
                 modelN{wnd}{layer}=mTRFmodelLoader(respN,stim{wnd}{layer}, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs);
            end
        end
        %}
end       
        
        
%     else
%         
%         dir_name=['../Data/',model_name,'/',mn,AllChn,Clean_name,sbj,'AllandGPT2FN(NoiseOnly_noAdapt',NpostAdapt,')_state_',num2str(stimState),'_lags', num2str(tmin), num2str(tmax),'zsc',num2str(zscoring),hannN,'HG_SpeechRE_PCA',num2str(ncmp),'.mat'];
%         model=mTRFmodelLoader(resp,STIM{stimState}, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs);
%             % loading and saving the files for noise
%         dir_name=['../Data/',model_name,'/',mn,AllChn,I_name,sbj,'AllandGPT2FN(NoiseOnly_noAdapt',NpostAdapt,')',NsameStim,'_state_',num2str(stimState),'_lags', num2str(tmin), num2str(tmax),'zsc',num2str(zscoring),hannN,'HG_SpeechRE_PCA',num2str(ncmp),'.mat'];
% 
%         if sameStim==0
%              modelN=mTRFmodelLoader(respN,STIMN{stimState}, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs);
%         elseif sameStim==1
%              modelN=mTRFmodelLoader(respN,STIM{stimState}, dir_name,load_prev_dec, lambdas, tmin, tmax,map,Fs);
%         end
%         
%     end
%     
%    Model{stimState}=model;
%    ModelN{stimState}=modelN;
%    clear model modelN
%     
% end
    



    %% predicting the signals with the trained models.
% for stimState=1:3
    if Manuall_Fix_Weight_edge==1
          for wnd=1:length(stimT)
                if wnd>1

                    for layer=1:nLYR
                        for runs = RUNS
                            %     win = hanning(7).^2;
                            %     w1 = repmat(win(1:4),
                            w = model{wnd}{layer}(runs).w;
                            w(:,1:3,:)=w(:,1:3,:).*.1; w(:,end-2:end,:)=w(:,end-2:end,:).*.1;
                            model{wnd}{layer}(runs).w = w;

                            %{
                            w = modelN{wnd}{layer}(runs).w;
                            w(:,1:3,:)=w(:,1:3,:).*.1; w(:,end-2:end,:)=w(:,end-2:end,:).*.1;
                            modelN{wnd}{layer}(runs).w = w;
                            %}

                        end
                    end
                else

                    for runs = RUNS
                        %     win = hanning(7).^2;
                        %     w1 = repmat(win(1:4),
                        w = model{wnd}(runs).w;
                        w(:,1:3,:)=w(:,1:3,:).*.1; w(:,end-2:end,:)=w(:,end-2:end,:).*.1;
                        model{wnd}(runs).w = w;


                    end

                end
          end
        
        
    end

    %predict
%     if stimState>1
      for wnd=1:length(stimT)        
        if wnd>1 % represents stim state
            for layer=1:nLYR
                for runs=RUNS
            %             model(runs).dir=model(runs).Dir;
            %             modelN(runs).dir=modelN(runs).Dir;
                    [pred{wnd}{layer}{runs},statsC{wnd}{layer}(runs)]=mTRFpredict(stimT{wnd}{layer}{runs},resp{runs},model{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
                    %{
                    if sameStim==0
                    [predN{wnd}{layer}{runs},statsN{wnd}{layer}(runs)]=mTRFpredict(stimN{wnd}{layer}{runs},respN{runs},modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
                    else
                    [predN{wnd}{layer}{runs},statsN{wnd}{layer}(runs)]=mTRFpredict(stim{wnd}{layer}{runs},respN{runs},modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
                    end
                    %}
                end
                wC = cat(4,model{wnd}{layer}.w);
    %             wN = cat(4,modelN{wnd}{layer}.w);
                rC = cat(1,statsC{wnd}{layer}.r);
    %             rN = cat(1,statsN{wnd}{layer}.r);
                allrC{wnd}{layer} = rC';
    %             allrN{wnd}{layer} = rN';
                allwC{wnd}{layer} = wC;
    %             allwN{wnd}{layer} = wN;
            end
        else
                for runs=RUNS
            %             model(runs).dir=model(runs).Dir;
            %             modelN(runs).dir=modelN(runs).Dir;
                    [pred{wnd}{runs},statsC{wnd}(runs)]=mTRFpredict(stimT{wnd}{runs},resp{runs},model{wnd}(runs));%,Fs,map,tmin,tmax);   %Test each model
                    %{
                    if sameStim==0
                    [predN{wnd}{layer}{runs},statsN{wnd}{layer}(runs)]=mTRFpredict(stimN{wnd}{layer}{runs},respN{runs},modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
                    else
                    [predN{wnd}{layer}{runs},statsN{wnd}{layer}(runs)]=mTRFpredict(stim{wnd}{layer}{runs},respN{runs},modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
                    end
                    %}
                end
                wC = cat(4,model{wnd}.w);
    %             wN = cat(4,modelN{wnd}{layer}.w);
                rC = cat(1,statsC{wnd}.r);
    %             rN = cat(1,statsN{wnd}{layer}.r);
                allrC{wnd} = rC';
    %             allrN{wnd}{layer} = rN';
                allwC{wnd} = wC;
    %             allwN{wnd}{layer} = wN;
            
        end
      end
%     else
%         
%         for runs=RUNS
%     %             model(runs).dir=model(runs).Dir;
%     %             modelN(runs).dir=modelN(runs).Dir;
%             [pred{stimState}{runs},statsC{stimState}(runs)]=mTRFpredict(STIM{stimState}{runs},resp{runs},Model{stimState}(runs));%,Fs,map,tmin,tmax);   %Test each model
%             if sameStim==0
%             [predN{stimState}{runs},statsN{stimState}(runs)]=mTRFpredict(STIMN{stimState}{runs},respN{runs},ModelN{stimState}(runs));%,Fs,map,tmin,tmax);   %Test each model
%             else
%             [predN{stimState}{runs},statsN{stimState}(runs)]=mTRFpredict(STIM{stimState}{runs},respN{runs},ModelN{stimState}(runs));%,Fs,map,tmin,tmax);   %Test each model
%             end
%         end
%         wC = cat(4,Model{stimState}.w);
%         wN = cat(4,ModelN{stimState}.w);
%         rC = cat(1,statsC{stimState}.r);
%         rN = cat(1,statsN{stimState}.r);
%         allrC{stimState} = rC';
%         allrN{stimState} = rN';
%         allwC{stimState} = wC;
%         allwN{stimState} = wN;
%         
%         
%     end
    
    
%     if stimState>1
      for wnd=1:length(stimT) 
          if wnd>1
                for shfl=1:nshfl
                    for layer=1:nLYR

                        for runs=RUNS

                            ttmax=min(size(stimT{wnd}{layer}{runs},1) , size(resp_sfl{runs,shfl},1));

                            [~,statsC_sfl(runs)]=mTRFpredict(stimT{wnd}{layer}{runs}(1:ttmax,:),resp_sfl{runs,shfl}(1:ttmax,:),model{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model

        %                     if sameStim==0
        %                     [~,statsN_sfl(runs)]=mTRFpredict(stimN{wnd}{layer}{runs}(1:ttmax,:),respN_sfl{runs,shfl}(1:ttmax,:),modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
        %                     else
        %                     [~,statsN_sfl(runs)]=mTRFpredict(stim{wnd}{layer}{runs}(1:ttmax,:),respN_sfl{runs,shfl}(1:ttmax,:),modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
        %                     end

                        end
                        rC_sfl=cat(1,statsC_sfl.r);
        %                 rN_sfl=cat(1,statsN_sfl.r);
                        allrC_sfl{wnd}{layer,shfl} = rC_sfl';
        %                 allrN_sfl{wnd}{layer,shfl} = rN_sfl';
                        clear statsC_sfl %statsN_sfl;
                    end
                end
          else
                for shfl=1:nshfl
                    

                        for runs=RUNS

                            ttmax=min(size(stimT{wnd}{runs},1) , size(resp_sfl{runs,shfl},1));

                            [~,statsC_sfl(runs)]=mTRFpredict(stimT{wnd}{runs}(1:ttmax,:),resp_sfl{runs,shfl}(1:ttmax,:),model{wnd}(runs));%,Fs,map,tmin,tmax);   %Test each model

        %                     if sameStim==0
        %                     [~,statsN_sfl(runs)]=mTRFpredict(stimN{wnd}{layer}{runs}(1:ttmax,:),respN_sfl{runs,shfl}(1:ttmax,:),modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
        %                     else
        %                     [~,statsN_sfl(runs)]=mTRFpredict(stim{wnd}{layer}{runs}(1:ttmax,:),respN_sfl{runs,shfl}(1:ttmax,:),modelN{wnd}{layer}(runs));%,Fs,map,tmin,tmax);   %Test each model
        %                     end

                        end
                        rC_sfl=cat(1,statsC_sfl.r);
        %                 rN_sfl=cat(1,statsN_sfl.r);
                        allrC_sfl{wnd}{1,shfl} = rC_sfl';
        %                 allrN_sfl{wnd}{layer,shfl} = rN_sfl';
                        clear statsC_sfl %statsN_sfl;
                    
                end
          end
      end
%     else
%         
%         for shfl=1:nshfl
% 
% 
%             for runs=RUNS
% 
%                 ttmax=min(size(STIM{stimState}{runs},1) , size(resp_sfl{runs,shfl},1));
% 
%                 [~,statsC_sfl(runs)]=mTRFpredict(STIM{stimState}{runs}(1:ttmax,:),resp_sfl{runs,shfl}(1:ttmax,:),Model{stimState}(runs));%,Fs,map,tmin,tmax);   %Test each model
%                 if sameStim==0
%                 [~,statsN_sfl(runs)]=mTRFpredict(STIMN{stimState}{runs}(1:ttmax,:),respN_sfl{runs,shfl}(1:ttmax,:),ModelN{stimState}(runs));%,Fs,map,tmin,tmax);   %Test each model
%                 else
%                 [~,statsN_sfl(runs)]=mTRFpredict(STIM{stimState}{runs}(1:ttmax,:),respN_sfl{runs,shfl}(1:ttmax,:),ModelN{stimState}(runs));%,Fs,map,tmin,tmax);   %Test each model
%                 end
% 
%             end
%             rC_sfl=cat(1,statsC_sfl.r);
%             rN_sfl=cat(1,statsN_sfl.r);
%             allrC_sfl{stimState}{shfl} = rC_sfl';
%             allrN_sfl{stimState}{shfl} = rN_sfl';
%             clear statsC_sfl statsN_sfl;
% 
%         end
%         
%         
%     end
% end
    %% 1 general model
    %{

    %align with time of start:and take average
    Model= model(1);
    tmp1=0;
    tmp2=0;
    for ii=1:length(model)
        tmp1=tmp1+model(ii).b;
        tmp2=tmp2+model(ii).w;
    end
    Model.w=tmp2/length(model);
    Model.b=tmp1/length(model);
    MODEL{total_C,noise}=Model;


    ModelN= modelN(1);
    tmp1=0;
    tmp2=0;
    for ii=1:length(modelN)
        tmp1=tmp1+modelN(ii).b;
        tmp2=tmp2+modelN(ii).w;
    end
    ModelN.w=tmp2/length(modelN);
    ModelN.b=tmp1/length(modelN);
    MODELN{total_C,noise}=ModelN;

    %}
label=[];
for run=1:length(outNew)
    label=[label;outNew(run).label];
end
%% calculate the correlations with a windowing method with zeropad:
%{
window_dur=700*1e-3;%500 mseconds
stepL=100*1e-3;% 10ms
Resp=zscore(vertcat(resp{:}));
Times=[];
TimesFlag=0;
for wnd=1:nWindow
    for layer=1:nLYR
        %Pred=vertcat(pred{wnd}{layer}{:});
        Pred=[];
        for pID=1:length(pred{wnd}{layer})
            Pred=[Pred;zscore(pred{wnd}{layer}{pID})];
        end
        nSteps=floor( (size(Pred,1)-window_dur*Fs)/(stepL*Fs) )+1;
        
        corrVals=[];
        for stp=1:nSteps
            if TimesFlag==0
                Times=[Times;floor((stp-1)*stepL*Fs+1)];
            end
            tminus=floor((stp-1)*stepL*Fs+1-ceil(window_dur*Fs/2));
            tplus=floor((stp-1)*stepL*Fs+ceil(window_dur*Fs/2));
            tin= max(tminus,1):min(tplus,size(Pred,1)) ;
            if tminus<1
                respTimes=[ zeros( 1-tminus, size(Resp(tin,:),2)); Resp(tin,:)];
                predTimes=[ zeros( 1-tminus, size(Pred(tin,:),2)); Pred(tin,:)];

            elseif tplus>size(Pred,1)
                respTimes=[ Resp(tin,:); zeros( tplus-size(Pred,1), size(Resp(tin,:),2))];
                predTimes=[ Pred(tin,:); zeros( tplus-size(Pred,1), size(Pred(tin,:),2))];

            else
                respTimes=Resp(tin,:);
                predTimes=Pred(tin,:);
            end
            tmp=[];
            for col=1:size(Pred,2)
                tmp=[tmp,corr(respTimes(:,col)+1e-6*rand(size(respTimes,1),1),predTimes(:,col)+1e-6*rand(size(predTimes,1),1))];
            end
            corrVals=[corrVals; tmp];
            %tOut= setxor(1:length(times), tIn);


        end
        TimesFlag=TimesFlag+1;
        allR_T{wnd}{layer}=corrVals;

            
       
        
    end
end
%}
    %%
rewrite=0;
add=['../Data/',mn,'_',sbjN,AllChn,gptname,'(',nameNcond,'_noAdapt',NpostAdapt,')',stpWrdN,NsameStim,nameWnd,'_allNAndSfl_cntD',num2str(contDvD),'_lag',num2str(tmin),'_', num2str(tmax),'zscore',num2str(zscoring),'hann',num2str(hannLengthB),edgeFixM,'_PCA',num2str(ncmp),'.mat'];
%add=['../Data/fw_AllandorGPT2F(NoiseOnly_noAdapt)_allNAndSfl_cntD1_lag0_700zscore2hann7_PCA20.mat'];
%add=['../Data/fw_AllandorGPT2F(NoiseOnly_noAdapt)_allNAndSfl_cntD1_lag0_700zscore2hann7_PCA20.mat'];
%add=['../Data/fw_GPT2FNW(NoiseOnly_noAdapt)_allNAndSfl_cntD1_lag0_700zscore2hann7_PCA20.mat'];
if rewrite==0 && isfile(add)
        load(add)
else

    if jjj==6
            save(add, 'allrC','allwC','elecs120','allrC_sfl','noiseNs','stimT' ,'resp','pred', 'Window','label','-v7.3') %'allR_T','Times','window_dur','stepL'
    else
            save(add, 'allrC','allwC','allrC_sfl','noiseNs','stimT' ,'resp','pred','Window','label','-v7.3')
    end
end

%%
% nLYR=2;
% Window=[0, 2 , 5, 10, 50, 100, 500, 3000];
% nWindow=length(Window);
%% estimate the msr for trials and each condintion:
% zscore diff trials of Resp and Pred to compare:
Resp=[];

for i=1:length(resp)
    resptmp=zscore(resp{i});
    Resp=[Resp;resptmp];
 
end

for wnd=1:length(stimT)
        for layer=1:nLYR
            Predtmp=[];
            for i=1:length(resp)
                if wnd>1
                predtmp=zscore(pred{wnd}{layer}{i});
                else
                predtmp=zscore(pred{wnd}{i});
                end
                Predtmp=[Predtmp;predtmp];
            end
            Pred{wnd}{layer}=Predtmp;
        end
end
WndMeth=0; % moving window or point wise diff
corrOverMsr=1;% correlation rather than msr
window_dur=500*1e-3;%500 mseconds
stepL=100*1e-3;% 10ms 
Times=[];
TimesFlag=0;
for wnd=1:length(stimT)
    for layer=1:nLYR
        if WndMeth==1 || corrOverMsr==1
            nSteps=floor( (size(Pred{wnd}{layer},1)-window_dur*Fs)/(stepL*Fs) )+1;
            msrVal=[];
            corrVals=[];
            for stp=1:nSteps
                if TimesFlag==0
                    Times=[Times;floor((stp-1)*stepL*Fs+1)];
                end
                tminus=floor((stp-1)*stepL*Fs+1-ceil(window_dur*Fs/2));
                tplus=floor((stp-1)*stepL*Fs+ceil(window_dur*Fs/2));
                tin= max(tminus,1):min(tplus,size(Pred{wnd}{layer},1)) ;
                if tminus<1
                    respTimes=[ zeros( 1-tminus, size(Resp(tin,:),2)); Resp(tin,:)];
                    predTimes=[ zeros( 1-tminus, size(Pred{wnd}{layer}(tin,:),2)); Pred{wnd}{layer}(tin,:)];

                elseif tplus>size(Pred{wnd}{layer},1)
                    respTimes=[ Resp(tin,:); zeros( tplus-size(Pred{wnd}{layer},1), size(Resp(tin,:),2))];
                    predTimes=[ Pred{wnd}{layer}(tin,:); zeros( tplus-size(Pred{wnd}{layer},1), size(Pred{wnd}{layer}(tin,:),2))];

                else
                    respTimes=Resp(tin,:);
                    predTimes=Pred{wnd}{layer}(tin,:);
                end
                if corrOverMsr==0
                    msrVal=[msrVal;mean( (predTimes-respTimes).^2 ,1)];
                else
                    tmp=[];
                    for col=1:size(Pred{wnd}{layer},2)
                        tmp=[tmp,corr(respTimes(:,col)+1e-6*rand(size(respTimes,1),1),predTimes(:,col)+1e-6*rand(size(predTimes,1),1))];
                    end
                    msrVal=[msrVal; tmp];
                end
            end
            allR_T{wnd}{layer}=msrVal;
        else
            Times=1:length(label);
            allR_T{wnd}{layer}= (Pred{wnd}{layer}-Resp).^2;
            stepL=10*10^-3;
        end
        TimesFlag=TimesFlag+1;
    end
end
%% time course of each noise:

PreOnset=floor(500*1e-3*Fs/(stepL*10^2));
PostOnset=floor(2000*1e-3*Fs/(stepL*10^2));
TC_avg_rC=zeros(nWindow,nLYR,4,PreOnset+PostOnset+1,size(allrC{2}{1},1));
TC_std_rC=zeros(nWindow,nLYR,4,PreOnset+PostOnset+1,size(allrC{2}{1},1));
for stimState=1:length(stimT)
    for layer=1:nLYR
        for noise=1:4
            [startT,stopT]=FindtimeSwitch(label(Times)==noise,1);
            TimeCourse=[];
            for tID=1:length(startT)
                TimeCourse=cat(3,TimeCourse,allR_T{stimState}{layer}(startT(tID)-PreOnset:startT(tID)+PostOnset,:));
            end
            TimeCourse_mean=mean(TimeCourse,3);
            TimeCourse_std=std(TimeCourse,[],3);
            TC_avg_rC(stimState,layer,noise,:,:)= TimeCourse_mean;
            TC_std_rC(stimState,layer,noise,:,:)= TimeCourse_std/sqrt(size(TimeCourse,3));
        end
    end
end
conditions={'spg', 'GPT', 'GPT+spg'};
layer=1;
figure()
for wnd=1:length(stimT)
    subplot(1,3,wnd)
    legendN=[];
    for noise=1:4
        data=squeeze(TC_avg_rC(wnd,layer,noise,:,:));
        data_std=squeeze(TC_std_rC(wnd,layer,noise,:,:));
        avg=mean(data,2);
        stes=sqrt(mean(data_std.^2, 2))./sqrt(size(data_std,2));
        x=1:length(avg);
        plot_shaded_error_bar(x,avg',stes',stes',noise,0,'-',0.1)
        hold on
        legendN=[legendN,{''},{noiseNs{noise}} ];
    end
legend(legendN)
xticks(1:50/(stepL*10^2):PostOnset+PreOnset+1)
xticklabels((-PreOnset:50/(stepL*10^2):PostOnset)/Fs*1000*(stepL*10^2))
xlabel('time( ms)')
title(['condition:', conditions{wnd}])
%  ylim([6,17]*1e-6)
%ylim([0.4,2.5])
end

%% all noise together
TC_avg_rC=zeros(nWindow,nLYR,2,PreOnset+PostOnset+1,size(allrC{2}{1},1));
TC_std_rC=zeros(nWindow,nLYR,2,PreOnset+PostOnset+1,size(allrC{2}{1},1));
for stimState=1:length(stimT)
    for layer=1:nLYR
            % clean
            [startT,stopT]=FindtimeSwitch(label(Times)==4,1);
            TimeCourse=[];
            for tID=1:length(startT)
                TimeCourse=cat(3,TimeCourse,allR_T{stimState}{layer}(startT(tID)-PreOnset:startT(tID)+PostOnset,:));
            end
            TimeCourse_mean=mean(TimeCourse,3);
            TimeCourse_std=std(TimeCourse,[],3);
            TC_avg_rC(stimState,layer,1,:,:)= TimeCourse_mean;
            TC_std_rC(stimState,layer,1,:,:)= TimeCourse_std/sqrt(size(TimeCourse,3));
            % all noise
            [startT,stopT]=FindtimeSwitch(label(Times)~=4 & label(Times)~=0,1);
            TimeCourse=[];
            for tID=1:length(startT)
                TimeCourse=cat(3,TimeCourse,allR_T{stimState}{layer}(startT(tID)-PreOnset:startT(tID)+PostOnset,:));
            end
            TimeCourse_mean=mean(TimeCourse,3);
            TimeCourse_std=std(TimeCourse,[],3);
            TC_avg_rC(stimState,layer,2,:,:)= TimeCourse_mean;
            TC_std_rC(stimState,layer,2,:,:)= TimeCourse_std/sqrt(size(TimeCourse,3));
    end
end

layer=2;
figure()
noiseNS={'clean','noise'};
for wnd=1:length(stimT)
    subplot(1,3,wnd)
    legendN=[];
    for noise=1:2
        data=squeeze(TC_avg_rC(wnd,layer,noise,:,:));
        data_std=squeeze(TC_std_rC(wnd,layer,noise,:,:));
        avg=mean(data,2);
        stes=sqrt(mean(data_std.^2, 2))./sqrt(size(data_std,2));
        x=1:length(avg);
        plot_shaded_error_bar(x,avg',stes',stes',noise,0,'-',0.2)
        hold on
        legendN=[legendN,{''},{noiseNS{noise}} ];
    end
legend(legendN)
xticks(1:50/(stepL*10^2):PostOnset+PreOnset+1)
xticklabels((-PreOnset:50/(stepL*10^2):PostOnset)/Fs*1000*(stepL*10^2))
xlabel('time( ms)')
title(['condition:', conditions{wnd}])
  %ylim([6,25]*1e-6)
%ylim([0.0,.3])
grid on
end

suptitle(['for layer=',num2str(layer)])
%%
%plot windows for noise and clean.
figure()
C=linspecer(nWindow,'sequential');
for noise=1:2
    subplot(1,2,noise)
    legendN=[];
    for wnd=1:length(stimT)
        data=squeeze(TC_avg_rC(wnd,layer,noise,:,:));
        data_std=squeeze(TC_std_rC(wnd,layer,noise,:,:));
        avg=mean(data,2);
        stes=sqrt(mean(data_std.^2, 2))./sqrt(size(data_std,2));
        x=1:length(avg);
        plot_shaded_error_bar(x,avg',stes',stes',wnd,0,'-',0.2,C)
        hold on
        legendN=[legendN,{''},{conditions{wnd}} ];
    end
    legend(legendN)
    title(noiseNS{noise})
xticks(1:50/(stepL*10^2):PostOnset+PreOnset+1)
xticklabels((-PreOnset:50/(stepL*10^2):PostOnset)/Fs*1000*(stepL*10^2))
    xlabel('time( ms)')
    %ylim([0.,0.5])
    grid on
end
suptitle(['for layer=',num2str(layer)])
%%  First: looking at the correlations:
sbj=1;
nWindow=length(stimT);
avg_cor_C=zeros(nWindow,nLYR,size(allrC{2}{1},1));
avg_cor_N=zeros(nWindow,nLYR,size(allrC{2}{1},1));
std_cor_C=zeros(nWindow,nLYR,size(allrC{2}{1},1));
std_cor_N=zeros(nWindow,nLYR,size(allrC{2}{1},1));
for stimState=1:length(stimT)
    for layer=1:nLYR
        % time of clean vs noisy
        [startT,stopT]=FindtimeSwitch(label(Times)==4,1);
        timesC=[];
        for inst=1:length(startT)
            timesC=[timesC; ((startT(inst) ):stopT(inst))'];%post_adapt*Fs
        end
        [startT,stopT]=FindtimeSwitch(label(Times)~=4 & label(Times)~=0,1);
        timesN=[];
        for inst=1:length(startT)
            timesN=[timesN; ((startT(inst) ):stopT(inst))'];%+post_adapt*Fs
        end
        
        avg_cor_C(stimState, layer,:)= mean(allR_T{stimState}{layer}(timesC,:),1);
        avg_cor_N(stimState, layer,:)= mean(allR_T{stimState}{layer}(timesN,:),1);   
        std_cor_C(stimState, layer,:)= std(allR_T{stimState}{layer}(timesC,:),[],1)/sqrt(length(timesC));
        std_cor_N(stimState, layer,:)= std(allR_T{stimState}{layer}(timesN,:),[],1)/sqrt(length(timesN)); 

    end
end
%{
figure()
legN=[];
for ss=1:nWindow
    yC=mean(avg_cor_C(ss,:,:),3);
    errC=sqrt(mean(std_cor_C(ss,:,:).^2, 3))/sqrt(size(std_cor_C,3));
    
    yN=mean(avg_cor_N(ss,:,:),3);
    errN=sqrt(mean(std_cor_N(ss,:,:).^2, 3))/sqrt(size(std_cor_N,3));
    x=1:nLYR;
    plot_shaded_error_bar(x,yC,errC,errC,ss,0,'-',0.15)
    hold on 
    plot_shaded_error_bar(x,yN,errN,errN,ss,0,'-.',0.08)
    hold on
    legN=[legN, {''},{num2str(Window(ss))}, {''},{num2str(Window(ss))}];
end
legend(legN )

xlabel('layer')
ylabel('corr')
title('average correlation over all electrodes, no adaptation')
%}
figure()
legN=[];
val_bar_C=[];
err_bar_C=[];
val_bar_N=[];
err_bar_N=[];
for ss=[1,3,2]
    yC=mean(avg_cor_C(ss,:,:),3);
    errC=sqrt(mean(std_cor_C(ss,:,:).^2, 3))/sqrt(size(std_cor_C,3));
    
    yN=mean(avg_cor_N(ss,:,:),3);
    errN=sqrt(mean(std_cor_N(ss,:,:).^2, 3))/sqrt(size(std_cor_N,3));
%     x=1:nLYR;
%     plot_shaded_error_bar(x,yC,errC,errC,ss,0,'-',0.2)
%     hold on 
%     plot_shaded_error_bar(x,yN,errN,errN,ss,0,'-.',0.2)
%     hold on

val_bar_C=[val_bar_C,yC'];
err_bar_C=[err_bar_C,errC'];
val_bar_N=[val_bar_N,yN'];
err_bar_N=[err_bar_N,errN'];
%     legN=[legN, {''},{num2str(Window(ss))}, {''},{num2str(Window(ss))}];
    legN=[legN,{conditions{ss}}];
end
h=bar_plot_sideBside_er_noX(val_bar_N,err_bar_N,0.2);
legend(legN )
hatchfill2(h)
% legend('','spgC','','spgN','','bertC','','bertN','','spg+BertC','','spg+BertN' )
hold on 
h=bar_plot_sideBside_er_noX(val_bar_C,err_bar_C,0.2);

xlabel('layer')
ylabel('corr')
grid on
title('average correlation over all electrodes, no adaptation (clean is hashed)')
%%
%findig electrodes that are significantly better:
layer=2;
% impr over the first window
ref=zeros(nLYR, size(allrC{wnd}{layer},1),size(allrC{wnd}{layer},2));
P_vals_1=zeros( nLYR, size(allrC{wnd}{layer},1));
refW=1;
trgW=2;
    for layer=1:nLYR
        for el=1:size(allrC{wnd}{layer},1)
            
                ref=allrN{refW}{layer}(el,:);
           
                tmp=allrN{trgW}{layer}(el,:);
                nshfl=1000;
                P_vals_1( layer,el)=RandomizationTest(tmp',ref',nshfl,true);

           
        end
    end


b=[squeeze(P_vals_1(1,:));squeeze(P_vals_1(2,:))];
disp('-------')
Sig_impr=sum(b<0.05,2)
Sig_deter=sum(b>0.95,2)
%%
% diff impr p-value
            
P_vals_diff=zeros(nWindow-1, nLYR, size(allrC{wnd}{layer},1));
for wnd=2:nWindow
    for layer=1:nLYR
        for el=1:size(allrC{wnd}{layer},1)
            
                ref=allrC{wnd-1}{layer}(el,:);
                tmp=allrC{wnd}{layer}(el,:);
                nshfl=10000;
                P_vals_diff(wnd-1, layer,el)=RandomizationTest(tmp',ref',nshfl,true);

           
        end
    end
end 
%%
layer=13;
b=[];
for el=1:size(allrC{wnd}{layer},1)
            
ref=allrN{1}{layer}(el,:);
tmp=allrN{6}{layer}(el,:);
                nshfl=10000;
                b=[b,[RandomizationTest(tmp',ref',nshfl,true); mean(tmp)-mean(ref)]];
end

%% looking at the improvement when context is added for clean vs noise:
layer=13;
y=mean(avg_cor_C(:,layer,:),3);
yE=sqrt(mean((squeeze(avg_cor_C(:,layer,:))-y).^2, 2))/sqrt(size(avg_cor_C,3));
plot_shaded_error_bar(1:size(y), y', yE',yE')
hold on


y=mean(avg_cor_N(:,layer,:),3);
yE=sqrt(mean((squeeze(avg_cor_N(:,layer,:))-y).^2, 2))/sqrt(size(avg_cor_N,3));
plot_shaded_error_bar(1:size(y), y', yE',yE')
hold on
%% C-N


% looking at the drop of correlations:

avg_cor_CN=zeros(3,nLYR,size(allrC{2}{1},1));
std_cor_CN=zeros(3,nLYR,size(allrC{2}{1},1));
for stimState=1:nWindow
    for layer=1:nLYR
  
        avg_cor_CN(stimState, layer,:)= mean(allrC{stimState}{layer}-allrN{stimState}{layer},2); 
        std_cor_CN(stimState, layer,:)= std(allrC{stimState}{layer}-allrN{stimState}{layer},[],2);
 
    end
end

figure()
for ss=1:3
    yC=mean(avg_cor_CN(ss,:,:),3);
    errC=sqrt(mean(std_cor_CN(ss,:,:).^2, 3))/sqrt(size(std_cor_CN,3));
    
    x=1:nLYR;
    plot_shaded_error_bar(x,yC,errC,errC,ss,0,'-',0.15)
    hold on 
end
legend('','spgC-N','','GPT2C-N','','spg+GPT2C-N' )

xlabel('layer')
ylabel('corr')
title('average correlation C-N over all electrodes')


%% looking at the change in correlation for each electrode when we move to next layers:
avg_corD_C=zeros(2,nLYR-1,size(allrC{2}{1},1));
avg_corD_N=zeros(2,nLYR-1,size(allrC{2}{1},1));
std_corD_C=zeros(2,nLYR-1,size(allrC{2}{1},1));
std_corD_N=zeros(2,nLYR-1,size(allrC{2}{1},1));
for stimState=2:3 % there is no diff for spg
    for layer=1:nLYR-1
        
        avg_corD_C(stimState-1, layer,:)= mean(allrC{stimState}{layer+1}-allrC{stimState}{layer},2);
        avg_corD_N(stimState-1, layer,:)= mean(allrN{stimState}{layer+1}-allrN{stimState}{layer},2);   
        std_corD_C(stimState-1, layer,:)= std(allrC{stimState}{layer+1}-allrC{stimState}{layer},[],2);
        std_corD_N(stimState-1, layer,:)= std(allrN{stimState}{layer+1}-allrN{stimState}{layer},[],2); 

    end
end

figure()
for ss=1:2
    yC=mean(avg_corD_C(ss,:,:),3);
    errC=sqrt(mean(std_corD_C(ss,:,:).^2, 3))/sqrt(size(std_corD_C,3));
    
    yN=mean(avg_corD_N(ss,:,:),3);
    errN=sqrt(mean(std_corD_N(ss,:,:).^2, 3))/sqrt(size(std_corD_N,3));
    x=1:nLYR-1;
    plot_shaded_error_bar(x,yC,errC,errC,ss,0,'-',0.15)
    hold on 
    plot_shaded_error_bar(x,yN,errN,errN,ss,0,'-.',0.08)
    hold on
end
legend('','GPT2C','','GPT2N','','spg+GPT2C','','spg+GPT2N' )

xlabel('layer')
ylabel('corr diff')
title('average correlation diff for two layers over all electrodes')

%% looking at some brain  plots for correlations:

% make sure sbj corresponds to the order of sbj
sbjLocationLoader;

%% looking at the  correlations for each reagon (STG, HG, and,,) for clean and noise
switch sbj
    case 6
        Rigions=[{'HG'} , {'PT'} , {'STG'},  {'SubcentralGyrus'}];
        nsbplt=2;
    case 5
        Rigions=[{'HG'} , {'Other'} , {'STG'},  {'insula'}];
        nsbplt=2;
    case 4
        Rigions=[{'HG'} , {'TTS'} , {'STG'},  {'insula'}];
        nsbplt=2;
    case 3
        Rigions=[{'HG'} , {'TTS'} , {'STG'},  {'insula'}];
        nsbplt=2;
    case 2
        Rigions=[{'MTG'} , {'Other'} , {'STG'}];
        nsbplt=2;
    case 1
        Rigions=[{'HG'} , {'TTS'} , {'STG'},  {'insula'}, {'STS'},{'supramarginal gyrus'}];
        nsbplt=3;
    otherwise
        
end
figure()

for i=1: length(Rigions)

    subplot(nsbplt, 2,i)
for ss=1:nWindow
    yC=mean(avg_cor_C(ss,:,strcmp(elecArea,Rigions{i})),3);
    errC=sqrt(mean(std_cor_C(ss,:,strcmp(elecArea,Rigions{i})).^2, 3))/sqrt(size(std_cor_C(2,:,strcmp(elecArea,Rigions{i})),3));
    
    yN=mean(avg_cor_N(ss,:,strcmp(elecArea,Rigions{i})),3);
    errN=sqrt(mean(std_cor_N(ss,:,strcmp(elecArea,Rigions{i})).^2, 3))/sqrt(size(std_cor_N(2,:,strcmp(elecArea,Rigions{i})),3));
    x=1:nLYR;
    plot_shaded_error_bar(x,yC,errC,errC,ss,0,'-',0.15)
    hold on 
    plot_shaded_error_bar(x,yN,errN,errN,ss,0,'-.',0.08)
    hold on

end
legend(legN )
ylim([0.1,0.35])
xlabel('layer')
ylabel('corr')
title(['average correlation over ',num2str(size(std_cor_N(2,:,strcmp(elecArea,Rigions{i})),3)),' electrodes in ',Rigions{i}] )
hold off
end
%%

figure()
for i=1: length(Rigions)
  subplot(nsbplt, 2,i)

for ss=1:3
    yC=mean(avg_cor_CN(ss,:,strcmp(elecArea,Rigions{i})),3);
    errC=sqrt(mean(std_cor_CN(ss,:,strcmp(elecArea,Rigions{i})).^2, 3))/sqrt(size(std_cor_CN(2,:,strcmp(elecArea,Rigions{i})),3));
    
    x=1:nLYR;
    plot_shaded_error_bar(x,yC,errC,errC,ss,0,'-',0.15)
    hold on 
    
end
legend('','spgC-N','','GPT2C-N','','spg+GPT2C-N' )
ylim([0,0.5])
xlabel('layer')
ylabel('corr')
title(['average correlation C-N over ',num2str(size(std_cor_N(2,:,strcmp(elecArea,Rigions{i})),3)),' electrodes in ',Rigions{i}] )
hold off
end

conditions={'spg', 'GPT2','spg+GPT2'};
for ss=1:3
figure()
C=linspecer(nLYR);
for i=1: length(Rigions)
subplot(nsbplt, 2,i)
hold on
nameLayer=[];

for layer=1:nLYR
    x=squeeze(avg_cor_C(ss,layer,strcmp(elecArea,Rigions{i})));
    y=squeeze(avg_cor_N(ss,layer,strcmp(elecArea,Rigions{i})));
    %scatter(x, y, 'filled'   ) 
    a = x\y;
    plot(x,a*x, 'color',C(layer,:))
    %plot(x,y,'*',x,a*x)
    nameLayer=[nameLayer;{num2str(layer)}];
end
xlim([min(avg_cor_C(ss, :,strcmp(elecArea,Rigions{i})),[],'all'), max(avg_cor_C(ss,:,strcmp(elecArea,Rigions{i})),[],'all')])
%ylim([min(avg_cor_N(ss,:,strcmp(elecArea,Rigions{i})),[],'all'), max(avg_cor_N(ss,:,strcmp(elecArea,Rigions{i})),[],'all')])
ylim([0,0.5])
line('LineStyle','--')
legend(nameLayer)
hold off
xlabel('corr clean')
ylabel('corr Noise')
title(Rigions{i})
end
suptitle(conditions{ss})
end


% from layer to layer improvement:
figure()

for i=1: length(Rigions)
  subplot(nsbplt, 2,i)
  
for ss=1:2
    yC=mean(avg_corD_C(ss,:,strcmp(elecArea,Rigions{i})),3);
    errC=sqrt(mean(std_corD_C(ss,:,strcmp(elecArea,Rigions{i})).^2, 3))/sqrt(size(std_corD_C(2,:,strcmp(elecArea,Rigions{i})),3));
    
    yN=mean(avg_corD_N(ss,:,strcmp(elecArea,Rigions{i})),3);
    errN=sqrt(mean(std_corD_N(ss,:,strcmp(elecArea,Rigions{i})).^2, 3))/sqrt(size(std_corD_N(2,:,strcmp(elecArea,Rigions{i})),3));
    x=1:nLYR-1;
    plot_shaded_error_bar(x,yC,errC,errC,ss,0,'-',0.15)
    hold on 
    plot_shaded_error_bar(x,yN,errN,errN,ss,0,'-.',0.08)
    hold on
end
legend('','GPT2C','','GPT2N','','spg+GPT2C','','spg+GPT2N' )
ylim([-0.1,0.1])
xlabel('layer')
ylabel('corr diff')
title(['average correlation of two layers over ',num2str(size(std_cor_N(2,:,strcmp(elecArea,Rigions{i})),3)),' electrodes in ',Rigions{i}] )

end
 
%% looking at clean corr of each layer brain plots
for layer=1:nLYR
    sbjName='LIJ120';
    Title=['elec corrs for layer',num2str(layer)];
    Val=avg_cor_C(:,layer);
    minV=min(Val);
    maxV=max(Val);
    PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)
end

%% looking at the layer to layer improvement brain plot

layer=12;% 4-> 5
sbjName='LIJ120';
Title=['elec corrs impr C from layer ',num2str(layer),' to ', num2str(layer+1)];
Val=avg_corD_C(:,layer);
minV=min(Val);
maxV=max(Val);
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)


sbjName='LIJ120';
Title=['elec corrs impr N from layer ',num2str(layer-1),' to ', num2str(layer)];
Val=avg_corD_N(:,layer);
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)

%% comparing layer 13 to 3
d_layer=13;
s_layer=3;
sbjName='LIJ120';
Title=['elec corrs layer',num2str(d_layer),' minus layer',num2str(s_layer)];
Val=avg_cor_C(:,d_layer)-avg_cor_C(:,s_layer);
minV=min(Val);
maxV=max(Val);
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)

%% comparing clean and noise for each layer:

layer=3;
sbjName='LIJ120';
Title=['elec corrs 1-(C-N) layer',num2str(layer)];
Val=1-(avg_cor_C(:,layer)-avg_cor_N(:,layer));
minV=min(Val);
maxV=max(Val);
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)

layer=13;
sbjName='LIJ120';
Title=['elec corrs 1-(C-N) layer',num2str(layer)];
Val=1-(avg_cor_C(:,layer)-avg_cor_N(:,layer));
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)


d_layer=13;
s_layer=3;
sbjName='LIJ120';
Title=['elec C-N corrs layer',num2str(s_layer),' minus that of layer',num2str(d_layer)];
Val=avg_cor_CN(:,s_layer)-avg_cor_CN(:,d_layer);
minV=min(Val);
maxV=max(Val);
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)




%% looking at the weights:
Latency_C=zeros(nLYR,size(allwC{1},3));
Latency_N=zeros(nLYR,size(allwC{1},3));
for layer=1:nLYR
    tmp_C=mean(allwC{layer},4);
    tmp_N=mean(allwN{layer},4);
    for elec=1:size(allwC{1},3)
       [~, imax]=max(squeeze( mean(tmp_C(1,:,elec).^2, 1)));
       Latency_C(layer,elec)=imax*10;% in ms
       [~, imax]=max(squeeze( mean(tmp_N(1,:,elec).^2, 1)));
       Latency_N(layer,elec)=imax*10;% in ms
    end
end

inp=[mean(Latency_C,2) mean(Latency_N,2) ];
errC=std(Latency_C,[], 2)/sqrt(size(Latency_C,2));
errN=std(Latency_N,[], 2)/sqrt(size(Latency_N,2));
inp_er=[ errC errN];
figure()
bar_plot_sideBside_er_noX(inp,inp_er); 
xlabel('layers')
ylabel('latency(ms)')
title(' average latencies over all electrodes for the forward model')


layer=12;
sbjName='LIJ120';
Title=['elec latency C',num2str(layer)];
Val=Latency_C(layer,:);
minV=min(Val);
maxV=max(Val);
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)


layer=12;
sbjName='LIJ120';
Title=['elec latency C-N',num2str(layer)];
Val=Latency_C(layer,:)-Latency_N(layer,:);
minV=min(Val);
maxV=max(Val);
PlotAvgBrain( Val, Ch2, Coor, Side, Title,[],minV,maxV, sbjName)

figure()
subplot(2,1,1)
imagesc(Latency_C)
subplot(2,1,2)
imagesc(Latency_N)
%% loom at each layer and the elec*lag map
layer=13;
figure()

pc_show=4;% pc+1
PC_strt=55;
for pc=1:pc_show % pc+1
% mean of all PCs:
if pc==1
    tmp_C=mean(allwC{layer},4);
    tmp_N=mean(allwN{layer},4);
    ElecLag_C=squeeze( mean(tmp_C(:,:,:).^2, 1))';
    ElecLag_N=squeeze( mean(tmp_N(:,:,:).^2, 1))';
    subplot(pc_show,2,1)

    tree=linkage(ElecLag_C,'ward'); % ward is for shortest ecludiane distance
    D = pdist(ElecLag_C);
    leafOrder = optimalleaforder(tree,D);
    minV=min(ElecLag_C(:));
    maxV=max(ElecLag_C(:));
    dendrogram(tree,size(ElecLag_C,1),'reorder',leafOrder,'Orientation','left' )
    imagesc(ElecLag_C(leafOrder,:),[ minV maxV])
    title('weight of clean, average of all PCs')

    yticks(1:length(elecArea))
    yticklabels(elecArea(leafOrder))
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',8)


    subplot(pc_show,2,2)
    imagesc(ElecLag_N(leafOrder,:),[ minV maxV])
    title('weight of noisy, average of all PCs')
    yticks(1:length(elecArea))
    yticklabels(elecArea(leafOrder))
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',8)
else

    ElecLag_C=squeeze( mean(tmp_C(pc+PC_strt-1,:,:).^2, 1))';
    ElecLag_N=squeeze( mean(tmp_N(pc+PC_strt-1,:,:).^2, 1))';
    subplot(pc_show,2,(pc-1)*2+1)

    minV=min(ElecLag_C(:));
    maxV=max(ElecLag_C(:));
    dendrogram(tree,size(ElecLag_C,1),'reorder',leafOrder,'Orientation','left' )
    imagesc(ElecLag_C(leafOrder,:),[ minV maxV])
    title(['weight of clean, average of all PC ',num2str(pc+PC_strt-1)])

    yticks(1:length(elecArea))
    yticklabels(elecArea(leafOrder))
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',8)

    subplot(pc_show,2,(pc-1)*2+2)
    imagesc(ElecLag_N(leafOrder,:),[ minV maxV])
    title(['weight of noisy, average of all PC ',num2str(pc+PC_strt-1)])
    yticks(1:length(elecArea))
    yticklabels(elecArea(leafOrder))
    a = get(gca,'YTickLabel');
    set(gca,'YTickLabel',a,'FontName','Times','fontsize',8)
end

end


suptitle('GPT2+spg')


%% TODO:
%{
1-looking at the correlations ( compare the corrs for clean and noise at
diffe layers) 
2-Looking at the latiencies ( compare the latencies at diff layers) ( look
at the word freq latencies)
3- look at the sorted neurons and see what neurons brain cares about. 
4- compare the latencies and correlations in noise
3-compare the latencies and correllations of phon etic features to the non
phonetic and see if when they doo bad the other ones doe better or not.

5- look at the similarity  ( adjust for anisotropy)
%}
%% looking at the corrs:
CorC=zeros(size(allrC{1},1), length(allrC));
CorN=zeros(size(allrC{1},1), length(allrC));
for layer=1:length(allrC)
    CorC(:,layer)=mean(allrC{layer},2);
    CorN(:,layer)=mean(allrN{layer},2);
end

figure()
for i=1:nLYR
    subplot(2,7,i)
    h=histogram(CorC(:,i));
    hold on
    histogram(CorN(:,i),h.BinEdges)
    xlabel('corr')
    legend('clean','noise')
    ylabel('histogram')
    title(['layer: ', num2str(i-1)])
    xlim([-0.5,0.6])
end
suptitle('histogram of corr for clean vs noise')
%% overal correlatin drop:

bar( mean((CorC-CorN), 1) )
xlabel('layer')
ylabel('diff in corr')
title(' avg of diff in corr (C-N)')
figure()
bar( mean((CorC-CorN)./CorC, 1) )
ylabel('rel diff in corr')
xlabel('layer')
title(' avg of rel diff in corr (C-N)')
%% PC correlations:

PCAs=[];
for layer=1:nLYR
    [~,PCA]=PCA_object(Stim{layer}, [],10);
    PCAs=[PCAs, {PCA}];
end

corrPCA_C=[];
corrPCA_N=[];
for layer=1:nLYR
    Astm=PCA_object(Stim{layer}, PCAs{layer});
    Aprd=PCA_object(Pred{layer}, PCAs{layer});
    AprdN=PCA_object(PredN{layer}, PCAs{layer});
    corrPCA_C=[corrPCA_C, diag(corr(Astm,Aprd))];
    corrPCA_N=[corrPCA_N, diag(corr(Astm,AprdN))];
end
%%
figure()
PCN=4;
for i=1:PCN
subplot(PCN,1,i)
bar([corrPCA_C(i,:); corrPCA_N(i,:)]')
title('corr of PCs in clean')
xlabel('layer')
legend('clean','noise')
title(['corr of PC:',num2str(i)])
ylabel('corr')
end


figure()
for i=1:PCN
subplot(PCN,1,i)
bar([corrPCA_C(i,:)- corrPCA_N(i,:)]')
title('corr of PCs in clean')
xlabel('layer')
legend('clean','noise')
title(['corr of PC:',num2str(i)])
ylabel('corr')
end

%% looking at the neuro activity of each layer (dendogram applied)
trl=1;
figure()
ID=1;
tt=1700:2700;
for layer=1:nLYR
    tree=linkage(stim{layer}{trl},'ward'); % ward is for shortest ecludiane distance
%     D = pdist(stim{layer}{trl});
%     leafOrder = optimalleaforder(tree,D);
    minV=min(Stim{layer}(:));
    maxV=max(Stim{layer}(:));
    subplot(3,5,ID);
%     
    [~,~,leafOrder]=dendrogram(tree,size(stim{layer}{trl},2),'Orientation','left' );
%     ylim([0.5 size(stim{layer}{trl},2)+0.5])
%     ax = gca;
%     set(ax,'XTick',[], 'YTick', []);
    imagesc(stim{layer}{trl}(tt,leafOrder)',[ minV maxV])
    set(gca,'YDir','normal')
    title([' GPT2 activity in layer ', num2str(layer)])
    colormap(jet)
    colorbar;
    
    minV=min(Pred{layer}(:));
    maxV=max(Pred{layer}(:));
    subplot(3,5,ID+5);
    imagesc(pred{layer}{trl}(tt,leafOrder)',[ minV maxV])
    set(gca,'YDir','normal')
    title([' pred, clean,  layer ', num2str(layer)])
    colormap(jet)
    colorbar;
    
    minV=min(PredN{layer}(:));
    maxV=max(PredN{layer}(:));
    subplot(3,5,ID+10);
    imagesc(predN{layer}{trl}(tt,leafOrder)',[ minV maxV])
    set(gca,'YDir','normal')
    title([' pred, noisy,  layer ', num2str(layer)])
    colormap(jet)
    colorbar;
    xlabel('t(ms)')
    
    ID=ID+1;
    if rem(layer,5)==0
        suptitle([' layers: ', num2str(layer-4),'-',num2str(layer)])
        figure()
        ID=1;
    end


end

%% looking at the cosine similarities:
figure()
tt=50000:60000;
for layer=1:nLYR
    
    cosSimC=getCosSim(Stim{layer}(tt,:),Pred{layer}(tt,:));
    cosSimN=getCosSim(Stim{layer}(tt,:),PredN{layer}(tt,:));
    cosSimC(cosSimC==0)=[];
    cosSimN(cosSimN==0)=[];
    x=1:length(cosSimC);
    subplot(2,7,layer)
    plot(x/Fs*10^3,cosSimC)
    hold on
    plot(x/Fs*10^3,cosSimN)
    hold off
    title(['layer ',num2str(layer)])
    legend('clean','noisy')
    xlabel('time(ms)')
end

%% looking at the average of similarities: without adjusting for isotrophy
avgSim=[];
stdSim=[];
for layer=1:nLYR
    
    cosSimC=getCosSim(Stim{layer},Pred{layer});
    cosSimN=getCosSim(Stim{layer},PredN{layer});
    cosSimC(cosSimC==0)=[];
    cosSimN(cosSimN==0)=[];
    avgSim=[avgSim; mean(cosSimC-cosSimN)];
    stdSim=[stdSim; std(cosSimC-cosSimN)];
end
 figure()
inp=avgSim;
% errC=std(dataC,0,1)/sqrt(size(dataC,1));
% errN=std(dataN,0,1)/sqrt(size(dataN,1));
inp_er=stdSim;%[ errC' errN'];
bar_plot_sideBside_er_noX(inp,inp_er); 

legend('clean','noise')

%% looking at the average of similarities: with adjusting for isotrophy

figure()
tt=50000:60000;
maxlag=2000*1e-3*Fs;% 700 ms
for layer=1:nLYR
    
    cosSimC=cross_cos_sim(Stim{layer}(tt,:),Pred{layer}(tt,:), maxlag);
    cosSimN=cross_cos_sim(Stim{layer}(tt,:),PredN{layer}(tt,:), maxlag);
    autoSim=cross_cos_sim(Stim{layer}(tt,:),Stim{layer}(tt,:), maxlag);
    x=-maxlag:maxlag;
    subplot(2,7,layer)
    yyaxis left
    plot(x/Fs*10^3,cosSimC)
    hold on
    plot(x/Fs*10^3,cosSimN)
    ylabel('clean and noise')
    yyaxis right
    plot(x/Fs*10^3,autoSim)
    ylabel('auto cosine sim')
    hold off
    title(['cross cosine sim, layer ',num2str(layer)])
    legend('clean','noisy','auto')
    xlabel('time(ms)')
end


%%
figure()
hold on
name=[];
for layer=1:nLYR
    plot(avg_cor_C')
    
    name=[name,{num2str(layer)}];
    
end
legend(name)
