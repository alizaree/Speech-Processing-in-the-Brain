function [new_signal] = conv_hamming(signal,w)
%convolves a sparse signal by a hamming window with length w, signal if
%matric the convolution is done on the columns of the signal
%   Detailed explanation goes here
if w~=0
    [~,c]=size(signal);
    new_signal=[];
    ham_w=hamming(w)';
    for i=1:c
        new_signal=[new_signal, conv(signal(:,i),ham_w,'same')];
    end
else
    new_signal=signal;
end
end

