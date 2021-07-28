function [out] = AddNewFeatures(in,phn, sem, newF,BF, GPT2, zscoring, PCA, n_spgCh)
% PCA yet to be determend
if ~exist('zscoring','var') || isempty(zscoring)
    zscoring=1;
end
if ~exist('PCA','var') || isempty(PCA)
    PCA=0;
end
if ~exist('n_spgCh','var') || isempty(n_spgCh)
    n_spgCh=16;
end
if ~exist('BF','var') || isempty(BF)
    addBF=0;
else
    addBF=1;
end

if ~exist('GPT2','var') || isempty(GPT2)
    addGPT=0;
else
    addGPT=1;
end

rm_fields=fieldnames(in);
index = strcmp(rm_fields, 'resp'); % this ensures that resp is zscored
rm_fields(index)=[];
if ~isempty(phn)
    %% first adding the aud Aud spec features
    loadload;
    close;

    for i=1:length(in)
        sz= size(in(i).resp,2);
        %Down_Sample= out(i).soundf/fs_after_DS; %put 1 if don't want to down sample
        tmp=resample(in(i).sound,16000,11025); %upsample the code
        sift=log2(16000/16000);
        tmp2=wav2aud(tmp,[10 10 -2 sift]);
        tmp4=tmp2';
        tmp4=tmp4(:,1:sz);
        tmp4=tmp4'.^0.33;%downsample(tmp4.^0.33, Down_Sample);
        tmp4 = tmp4(:,1:120);
        in(i).aud_sound= resample(tmp4',1,8)';

        %in(i).spg= spectrogram(x)
    end
    %% deviding them into 16 groups and adding the derivatives:
    %n_spgCh=size(in(1).aud_sound,2); % uncomment for full spg
    N_f_inCh=size(in(1).aud_sound,2)/n_spgCh;
    n_spgCh=size(in(1).aud_sound,2);
    for i=1:length(in)
        %{
        tmp=in(i).aud_sound;
        
        tmp2=[];
        for j=1:n_spgCh
            tmp2=[tmp2, sum(tmp(:, (j-1)*N_f_inCh+1:j*N_f_inCh), 2)];
        end
        
        in(i).aud_sound=tmp2;
        %}
        in(i).aud_D_sound=reluf([zeros(1,n_spgCh);diff(in(i).aud_sound)]);
        
    end

end
%% high pass filtering resp and taking the tanh
Fs=100;
timeWindow=25; % in seconds
cutOfFreq=.001; % in hz
[b,a] = fir1(timeWindow*Fs,cutOfFreq,'high');
figure()
freqz(b,a)
% concatinate the data for each electrode:
Resp=horzcat(in.resp);
RespN=[];
for el=1:size(Resp,1)
tmp2 = filtfilt(b,a,Resp(el,:));
RespN=[RespN;tmp2];

end

% tanh of 8 std of filtered output of resp
for el=1:size(RespN,1)
    stdV=std(RespN(el,:));
    RespN(el,:)=8*(stdV)*tanh( ( RespN(el,:)-mean(RespN(el,:)) )/(8*stdV) )+mean(RespN(el,:));
    
end

startL=1;
for i=1:length(in)
    in(i).resp=RespN(:,startL:startL+size(in(i).resp,2)-1);
    startL=startL+size(in(i).resp,2);
end   



%% second, adding the phn feature:
eeg_file_name=[];
for i=1:length(in)
    eeg_file_name=[eeg_file_name;{in(i).name}];
end
w_ph=0;%5*3; %samples
w_w=0;%14*3; %samples
if addGPT
    [logidx,idx]=ismember(eeg_file_name, GPT2.file_name);
elseif addBF
    [logidx,idx]=ismember(eeg_file_name, BF.file_name);
else
    [logidx,idx]=ismember(eeg_file_name, phn.file_name);
end
wind=[1,4,6,8];
    
for i=1:length(in)
    nz=size(in(i).resp,2);
    if logidx(i)
        if ~isempty(phn)
        in(i).ph_idx_tVec=VecZeroPad(phn.ph_idx_tVec{idx(i)},nz);
        in(i).ph_Vec_tVec=VecZeroPad(conv_hamming(phn.ph_Vec_tVec{idx(i)},w_ph),nz);
        in(i).phCohortReductVec=VecZeroPad(conv_hamming(phn.phCohortReductVec{idx(i)},w_ph),nz);
        in(i).phCohortVec=VecZeroPad(conv_hamming(phn.phCohortVec{idx(i)},w_ph),nz);
        in(i).phEntrVec=VecZeroPad(conv_hamming(phn.phEntrVec{idx(i)},w_ph),nz);
        in(i).phSurpVec=VecZeroPad(conv_hamming(phn.phSurpVec{idx(i)},w_ph),nz);
        in(i).words_onset=VecZeroPad(conv_hamming(phn.words_onset{idx(i)},w_w),nz);
        end
        if ~isempty(sem)
        in(i).sem_feature_Vec=VecZeroPad(conv_hamming(sem.semantic_Vec{idx(i)}.semantic_features_Vec,w_w),nz);
        in(i).sem_dis_Vec=VecZeroPad(conv_hamming(sem.semantic_Vec{idx(i)}.sem_dis_Vec,w_w),nz);
        end
        %new feartures
        if ~isempty(newF)
        in(i).ph_nVec_tVec=VecZeroPad(conv_hamming(newF.ph_Vec_tVec{idx(i)},w_ph),nz);
        in(i).ph_on=VecZeroPad(conv_hamming(newF.ph_on{idx(i)},w_ph),nz);
        in(i).ph_bigram_tVec=VecZeroPad(conv_hamming(newF.ph_bigram_tVec{idx(i)},w_ph),nz);
        in(i).cohort_entr_tVec=VecZeroPad(conv_hamming(newF.cohort_entr_tVec{idx(i)},w_ph),nz);
        in(i).cohort_surp_tVec=VecZeroPad(conv_hamming(newF.cohort_surp_tVec{idx(i)},w_ph),nz);
        in(i).word_on=VecZeroPad(conv_hamming(newF.word_on{idx(i)},w_ph),nz);
        in(i).word_f_tVec=VecZeroPad(conv_hamming(newF.word_f_tVec{idx(i)},w_ph),nz);
        in(i).sem_den_tVec=VecZeroPad(conv_hamming(newF.sem_den_tVec{idx(i)},w_ph),nz);
        end

        if addBF
            if length(size(BF.Bert_act{1}))==4
                in(i).bert_act=VecZeroPad(BF.Bert_act{idx(i)}(:,wind,:,:),nz);
            else
                in(i).bert_act=VecZeroPad(BF.Bert_act{idx(i)},nz);
            end
        end
        
        if addGPT
            in(i).GPT2_act=VecZeroPad(GPT2.GPT2_act{idx(i)},nz);
            in(i).nonStopWrds=VecZeroPad(GPT2.nonStopWrds{idx(i)},nz);
        end
    
    
    end
    in(i).resp=in(i).resp';
    
end

fields=fieldnames(in);
index = strcmp(fields, 'ph_idx_tVec'); % do not zscore ph_idx_tVec
fields(index)=[];
index = strcmp(fields, 'nonStopWrds'); % do not zscore 'nonStopWrds'
fields(index)=[];

index = strcmp(fields, 'GPT2_act'); % do not zscore 'GPT2_act'

fields(index)=[];

index = strcmp(fields, 'bert_act');% do not zscore 'bert_act

fields(index)=[];

added_fields=setdiff(fields,rm_fields);
if sum(strcmp(added_fields,'GPT2_act'))
    in=makeSingle(in,[], {'GPT2_act'});
end
if sum(strcmp(added_fields,'bert_act'))
    in=makeSingle(in,[], {'bert_act'});
end

% zscoring
if zscoring==1
    for j=1:length(added_fields)
        temp=[];
        for i=1:length(in)
            temp=cat(1,temp, in(i).(added_fields{j}) );
        end

        stdd.(added_fields{j})=std(temp,[],1);
        meann.(added_fields{j})=mean(temp,1);
    end

    for j=1:length(added_fields)
        for i=1:length(in)
            in(i).(added_fields{j})=bsxfun(@minus,in(i).(added_fields{j}), meann.(added_fields{j}));
            in(i).(added_fields{j})=bsxfun(@rdivide,in(i).(added_fields{j}), stdd.(added_fields{j}));
        end
    end
    
elseif zscoring==2 % normalize but keep the mean value untouched.

    for j=1:length(added_fields)
        if ~strcmp(added_fields{j}, 'GPT2_actjhvy')%'bert_act')%
            
            temp=[];
            for i=1:length(in)
                temp=cat(1,temp, in(i).(added_fields{j}) );
            end

            nrm.(added_fields{j})=vecnorm(temp,2,1);
        else
            if addGPT
                temp=[];
                for i=1:length(in)
                    temp=cat(1,temp, in(i).(added_fields{j})(:,1:8,:) );
                end
                nrm1=vecnorm(temp,2,1);

                temp=[];
                for i=1:length(in)
                    temp=cat(1,temp, in(i).(added_fields{j})(:,9:17,:) );
                end
                nrm2=vecnorm(temp,2,1);

                temp=[];
                for i=1:length(in)
                    temp=cat(1,temp, in(i).(added_fields{j})(:,18:end,:) );
                end
                nrm3=vecnorm(temp,2,1);

                nrm.(added_fields{j})=cat(2, nrm1,nrm2, nrm3);
            
            end
            
            if addBF
                temp=[];
                for i=1:length(in)
                    temp=cat(1,temp, in(i).(added_fields{j})(:,1,:,:) );
                end
                nrm1=vecnorm(temp,2,1);

                temp=[];
                for i=1:length(in)
                    temp=cat(1,temp, in(i).(added_fields{j})(:,2,:,:) );
                end
                nrm2=vecnorm(temp,2,1);

                temp=[];
                for i=1:length(in)
                    temp=cat(1,temp, in(i).(added_fields{j})(:,3,:,:) );
                end
                nrm3=vecnorm(temp,2,1);
                
                temp=[];
                for i=1:length(in)
                    temp=cat(1,temp, in(i).(added_fields{j})(:,4,:,:) );
                end
                nrm4=vecnorm(temp,2,1);

                nrm.(added_fields{j})=cat(2, nrm1,nrm2, nrm3,nrm4);
            end
            
            
            
        end
%     meann.(added_fields{j})=mean(temp,1);
    end

    for j=1:length(added_fields)
        for i=1:length(in)
%             in(i).(added_fields{j})=bsxfun(@minus,in(i).(added_fields{j}), meann.(added_fields{j}));
            in(i).(added_fields{j})=bsxfun(@rdivide,in(i).(added_fields{j}), nrm.(added_fields{j}));
        end
    end
    
    
end
out=in;
% add all
end

