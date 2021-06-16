
data = 'hello world';

mes= text2bin(data);

encode = blockcode52_encode(mes);

decode = blockcode52_decode(encode);

bin2text(decode)