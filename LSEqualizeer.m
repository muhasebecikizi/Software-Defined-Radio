
% LSequalizer.m find a LS equalizer f for the channel b
k=3; re = n; % length of equalizer - 1
delta=3; % use delay <=n*length(b) 
p=length(re)-delta;
RE=toeplitz(re(k+1:p),re(k+1:-1:1)); % build matrix R 
SE=re(k+1-delta:p-delta)'; % and vector SE
f=inv(RE'*RE)*RE'*SE % calculate equalizer 
Jmin=SE'*SE-SE'*RE*inv(RE'*RE)*RE'*SE % Jmin for this f and delta
ye=filter(f,1,re); % equalizer is a filter
dec=sign(ye); % quantize and find errors
err=0.5*sum(abs(dec(delta+1:end)-re(1:end-delta)))
z = ye;