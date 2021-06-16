

close all
clear all 
clc

% To check the all steps of the receiver signal find(receiver_text~=assumed text)
% function can be used 

% specification of impairments
cng = 0.6 %input('channel noise gain: try 0, 0.6 or 2 :: ');
cdi=0 %input('channel multipath: 0 for none, 1 for mild or 2 for harsh :: '); fo =input('tranmsitter mixer freq offset in %: try 0 or 0.01 :: ');*
po=0.7 %input('tranmsitter mixer phase offset in rad: try 0, 0.7 or 0.9 :: '); toper=input('baud timing offset as % of symb period: try 0, 20 or 30 :: '); so=input('symbol period offset: try 0 or 1 :: ');
so=0 %input('symbol period offset: try 0 or 1 ::  ');*
fo =0 %input('tranmsitter mixer freq offset in %: try 0 or 0.01 ::  ');*  
toper=0 %input('baud timing offset as % of symb period: try 0, 20 or 30 ::  ');
% 

% %TRANSMITTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 
% % encode text string as T-spaced PAM (+/-1, +/-3) sequence

% str2='01234 I wish I were an Oscar Mayer wiener 56789 ';
str3 = 'A0Oh well whatever Nevermindl ';

% str = 'Ada sahillerinde bekliyorum';
message=text2bin(str3); % change text into 7 bit binary using text2bin 
% % 7 * 11 = 77 bit for symbol (1*77 array)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 
% % encode
% 
% coded_txt = message;
coded_txt = blockcode52_encode(message);
% % 1*200 array 

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 
% % binery to 4-PAM
% 
j=1; mpam=zeros(1,ceil(length(coded_txt)/2));
for i=1:2:length(coded_txt)-1
if coded_txt(i:i+1)==[0,0], mpam(j)=-3; end 
if coded_txt(i:i+1)==[0,1], mpam(j)=-1; end 
if coded_txt(i:i+1)==[1,0], mpam(j)=1; end 
if coded_txt(i:i+1)==[1,1], mpam(j)=3; end 
j=j+1;
end
figure(1)
plot (mpam)

% %returns mpam 
% % 1*100 array
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 
 mpam_len=length(mpam);  % 4-level signal of length N % zero pad T-spaced symbol sequence to create upsampled T/M-spaced  
% sequence of scaled T-spaced pulses (with T = 1 time unit)
% 
M=200-so; 
mup=zeros(1,mpam_len*M); 
mup(1:M:end)=mpam; % oversampling factor 
%signal (1*100 array) multiplied with 200 => 1*20000 array
% 
% %SRRC pulse filter with T/M-spaced impulse response
% 
srrc_L=0.5;
p=(14.0)*srrc(srrc_L,0.3,M,0.4);
x=filter(p,1,mup); % oversamled and srrc signal

figure(2), plotspec(x,1/M) 
% 
% % output x

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%AM modulation
t_mod=1/M:1/M:length(x)/M; % T/M-spaced time vector

Fc=20; % carrier frequency 

c=cos(2*pi*(Fc*(1+0.01*fo))*t_mod+po); % carrier with offsets relative to rec osc 
r=c.*x; % modulate message with carrier 

figure(3), plot(r(1:10*M));
figure(4), plotspec(r,1/M);
figure(5), plotspec(p,1/M);
 
%output r

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%IMPAIRMENT : FADING

ds=pow(r); % desired average power of signal 
lr=length(r); % length of transmitted signal vector 
fp=[ones(1,floor(0.2*lr)),0.5*ones(1,lr-floor(0.2*lr))]; % flat fading profile
r=r.*fp; % apply profile to transmitted signal vector

if cdi < 0.5 % channel ISI
  mc=[1 0 0]; % distortion-free channel
elseif cdi<1.5
mc=[1 zeros(1,M) 0.28 zeros(1,2.3*M) 0.11]; % mild multipath channel
else
mc=[1 zeros(1,M) 0.28 zeros(1,1.8*M) 0.44]; % harsh multipath channel
end
mc=mc/(sqrt(mc*mc')); % normalize channel power
dv=filter(mc,1,r); % filter transmitted signal through channel
nv=dv+cng*(randn(size(dv))); % add Gaussian channel noise

to=floor(0.01*toper*M); % fractional period delay in sampler

rnv=nv(1+to:end); % delay in on-symbol designation
rt=(1+to)/M:1/M:length(nv)/M; % modified time vector with delayed message start
rM=M+so; % receiver sampler timing offset (delay)
% 
% rnv güncel sinyal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
%RECEIVER
% 
% %  Automatic Gain Control (AGC)
% 
% % % %this algorithm merges heuristic and gradient descent algorithm
% % % vr=pow(message) ;                % power of the input
AGC_mu=0.0003;                     % algorithm stepsize
% lenavg=1;                        % length over which to average, 
                                   %normalde avg gelen her bir 's' sample'na deðil 
                                   %avg alarak 'S' ile karþýlatýrma yapýlýyor, 
                                   %biz þimdilik samplelara tek tek bakarak 
                                   %karþýlaþtýran algoritmayý kullandýðýmýz için 
                                   %avg lengthini 1 olarak alýyoruz.
                                   
lenavg = 1;                         % length over which to average
AGC_gain=zeros(1,lr); 
AGC_gain(1)= 1.5;                  % initialize AGC parameter
nr=zeros(1,lr);                    % initialize outputs
avec=zeros(1,lenavg);              % vector to store terms for averaging
for k=1:lr-1
  nr(k)=AGC_gain(k)*rnv(k);                  % normalize by a(k)
%   avec=[sign(AGC_gain(k))*(nr(k)^2-ds),avec(1:lenavg-1)];  % J_N incorporate new update into avec
  avec = [(sign(nr(k))*(nr(k)^2 - ds)),avec(1:lenavg-1)]; %J_LS
  AGC_gain(k+1)= AGC_gain(k)-AGC_mu*mean(avec);       % average adaptive update of a(k)
end
figure(6)
plot(nr)
rnv = nr;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%% Heuristic AGC %%%%
% AGC_gain=zeros(1,lr); 
% AGC_gain(1)=1.5;                       % initialize AGC parameter
% nr=zeros(1,lr);                      % initialize outputs
% mu=0.0003;                           % algorithm stepsize
% for k=1:lr-1
%   nr(k)=AGC_gain(k)*rnv(k);                  % normalize by a to get s
%   AGC_gain(k+1)= AGC_gain(k) - mu*(nr(k)^2-ds);      % adaptive update of a(k)
% end
% figure(6)
% plot(nr)
% 
% rnv = nr;


rpll=rnv; % rsc is from pulrecsig.m
fl=100; % low pass filter frequency
ff=[0 .01 .02 1];
fa = [1 1 0 0];
h=remez(fl,ff,fa); % LPF design
pll_mu=.003; % algorithm stepsize
fc=20; % assumed freq. at receiver
theta=zeros(1,length(t_mod)); 
theta(1)=0; % initialize estimate vector
zs=zeros(1,fl+1); % initialize buffers for LPFs
zc=zeros(1,fl+1); % z's contain past fl+1 inputs

for k=1:length(t_mod)-1
    
    zs=[zs(2:fl+1), 2*rpll(k)*sin(2*pi*fc*t_mod(k)+theta(k))];
    zc=[zc(2:fl+1), 2*rpll(k)*cos(2*pi*fc*t_mod(k)+theta(k))];
    lpfs=fliplr(h)*zs'; 
    lpfc=fliplr(h)*zc'; % new output of filters
    theta(k+1)=theta(k)-pll_mu*lpfs*lpfc; %algorithm update  
end

figure(7),plot(t_mod,theta),
title('Phase Tracking via the Costas Loop')
xlabel('time'); 
ylabel('phase offset');

theta;
phoff = theta;
% % 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 
% % AM  demodulation of received signal sequence r
ram = rnv;
fc = 20;
c2=cos(2*pi*fc*t_mod + phoff);
x2=ram.*c2; % LPF design
fl=100;
fbe=[0 0.1 0.2 1];
damps=[1 1 0 0 ]; % design of LPF parameters 
b=remez(fl,fbe,damps); % create LPF impulse response 
x3=2*filter(b,1,x2); % LPF and scale downconverted signal 
figure(8),plotspec(x3,1/M) 
% % 
% % 
% % output x3
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 
% % matchfilt.m: test of SNR maximization
% 
% x3 = r; fl = 100;

recfilt=(15)*srrc(srrc_L,0.3,M,0.0); % receive filter H sub R
%recfilt=recfilt/sqrt(sum(recfilt.^2)); % normalize the pulse shape
v=1/180*filter(fliplr(recfilt),1,x3); % matched filter with data 
figure(9), 
ul=floor((length(v)-124)/(4*rM)); 
plot(reshape(v(125:ul*4*rM+124),4*rM,ul)) % plot the eye diagram 

% % output v
% plot(v)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % clock recovery algorithm
% 
xcl = v; 
clock_n =mpam_len ; 
clock_L = srrc_L; 
tnow=clock_L*M+1; 
tau=0; 
xs=zeros(1,clock_n);          % initialize variables
tausave=zeros(1,clock_n); 
tausave(1)=tau; 
i=0;
clock_mu=0.003;                                  % algorithm stepsize
delta=0.1;                                 % time for derivative
while tnow<length(xcl)-2*clock_L*M                 % run iteration
  i=i+1;
  xs(i)=interpsinc(xcl,tnow+tau,clock_L);          % interpolated value at tnow+tau
  x_deltap=interpsinc(xcl,tnow+tau+delta,clock_L); % get value to the right
  x_deltam=interpsinc(xcl,tnow+tau-delta,clock_L); % get value to the left
  dx=x_deltap-x_deltam;                    % calculate numerical derivative  
  qx=quantalph(xs(i),[-3,-1,1,3]);         % quantize xs to nearest 4-PAM symbol
  tau=tau+clock_mu*dx*(qx-xs(i));                % alg update: DD 
  tnow=tnow+M; tausave(i)=tau;             % save for plotting
end
figure(10), 
subplot(2,1,1), 
plot(xs(1:i-2),'b.')    % plot constellation diagram
title('constellation diagram');
ylabel('estimated symbol values')
subplot(2,1,2), 
plot(tausave(1:i-2))    % plot trajectory of tau
ylabel('timing offset estimates'), xlabel('iterations')
tausave;
% % 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  % downsample to symbol rate
% %  
% % 
z=v(0.5*fl+rM:rM+so:end);% set delay to first symbol-sample and increment by M 
figure(11), plot([1:length(z)],z,'*','MarkerSize',12) % soft decisions 
% %  
% % %z
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % LSequalizer.m find a LS equalizer f for the channel b
n=3; 
re = z; % z or mpam    %length of equalizer - 1 
LS_delta=0; % use delay <=n*length(b) 
p=length(re)
RE=toeplitz(re(n+1:p),re(n+1:-1:1)); % build matrix R 
SE=re(n+1-LS_delta:p-LS_delta)'; % and vector SE 
f=inv(RE'*RE)*RE'*SE % calculate equalizer f
Jmin=SE'*SE-SE'*RE*inv(RE'*RE)*RE'*SE % Jmin for this f and delta 
ye=filter(f,1,re); % equalizer is a filter 
dec=sign(ye); % quantize and find errors 
err=0.5*sum(abs(dec(LS_delta+1:end)-re(1:end-LS_delta)))
z = ye;

figure(12)
plot(z)
figure(13)
plot(mpam) %en baþtaki sinyal karþýlaþtýrmak için
length(z)

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % decision device and symbol matching performance assessment
% 
 mprime=quantalph(z,[-3,-1,1,3])'; % quantize to +/-1 and +/-3 alphabet 
 cluster_variance=(mprime-z)*(mprime-z)'/length(mprime), % cluster variance
 
j=1;
y=zeros(1,2*length(mprime)); 
rq=quantalph(mprime,[-3,-1,1,3]); 
for i=1:length(mprime)
if rq(i)==3
y(j:j+1)=[1,1]; 
end
if rq(i)==1
    y(j:j+1)=[1,0]; 
end
if rq(i)==-1
    y(j:j+1)=[0,1]; 
end
if rq(i)==-3 
    y(j:j+1)=[0,0];
end

j=j+2;

end

figure(14)
plot(y)
length(y)
% 
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% % % decode
% y1 = y;
y1 = blockcode52_decode(y);
% % plot(y1)
% % % figure
% % plot(message)
% % length(y1)
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % binary to text
ytext=bin2text(y1)
% length(ytext)
% length(str)
% numerr=length(find(ytext~=text)); % text
% lmp=length(mprime); % number of recovered symbol estimates 
% percentage_symbol_errors=100*sum(abs(sign(mprime-mpam(1:lmp))))/lmp % symb err 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%