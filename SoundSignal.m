clc;
clear all;


% quantization level
% For the quantization N level is 256 and operational
% bit number is 8 (2^8 = 256)
N = 8;


%Load the audio file

[y,Fs]=audioread('gong.wav');

Len=length(y); % total audio length

sound(y,Fs); % plays audio files
t = [0:1/Fs:(length(y)/Fs)-1/Fs]; % total audio length in seconds
mono_y = y(:,1); %stereo to mono conversion
% channel reduction to increase operational speed



figure(1)
subplot(3,1,1);
plot(t,mono_y);
title('Original Audio Signal');
xlabel('Time');
ylabel('Amplitude');

% Quantization
% level value(256) can be changed
% standart formula applied
% Analog signal amplitudes catogarized with 256 level
% Each amplitude represented with different level values
quantization = (max(mono_y)-(min(mono_y)))/(2^(N)) 
% result should be integer value
y_quantized = round(mono_y/quantization);


% Quantization error calculation
for i=1:Len
    errq(i)=y_quantized(i)*quantization-mono_y(i);
end

errquan=sum(errq.*errq)/Len
%stem(errorq)


%Plot quantized
subplot(3,1,2);
hold on
plot(t,mono_y);
stairs(t,y_quantized*(quantization)); 

title('Quantized Signal vs Original Signal');
xlabel('Time');
ylabel('Amplitude');


% positive and negative quantization levels are specified
% uniform quantization distrubiton used and 256 level
% distrubited on both negative and positive sides
sn =[zeros(Len,1)];

for i = 1 : Len
    if y_quantized(i)<=0
        sn(i)=1;
    else 
        sn(i)=2;
    end
end
  

  
  
% Quantization output converted to binary values (8 binary values)
% front of data added their positive or negative
% quantization values (1 or 0)
% totaly, 9 bit binary packages obtained
binaryo = [dec2bin(sn) dec2bin(abs(y_quantized),N)];




binary_array = zeros(1,(N+2)*Len);

% dec2bin results are string values 
% however binary values are needed
% Obtained packages work on ASCII code sheme
% on this step, ASCII codes converted to binary values.
for i = 1 : Len
    for j = 1 : (N+2)
        binary_array(((i-1)*(N+2)) + (j)) = binaryo(i,j)-48; 
    end
end

sample_of_signal = binary_array(1:10)
