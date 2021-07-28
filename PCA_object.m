function [new_data,PCA] = PCA_object(data, PCA, nComponents )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    % data nsample*nf
    if ~exist('nComponents','var') || isempty(nComponents)
        nComponents=PCA.nComponents;
    end
    if ~exist('PCA','var') || isempty(PCA)
        [data,mu,sigma]=zscore(data);
        PCA.mu=mu;
        PCA.sigma=sigma;
        PCA.nComponents=nComponents;
        a=data'*data;
        [v,s]=eig(a);
        d=size(a,1);
        ss=zeros(d,1);
        for i=1:d
            ss(i)=sum(s(i,:));
        end

        [~,idx]=sort(ss,'descend');
        transform=v(:,idx);
        PCA.transform=transform;
    else
        data=bsxfun(@rdivide, bsxfun(@minus, data, PCA.mu),PCA.sigma);
        nComponents=PCA.nComponents;
    end
    

    new_data=data*PCA.transform(:,1:nComponents);
end

