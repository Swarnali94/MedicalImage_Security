
clc;
clear all;
close all;
warning off; 

z=(imread('ultrasound of abdomen.jpg'));
xr=imresize(z,[512 512]);
 ximg=rgb2gray(xr);
 figure,imshow(ximg); title('Host Image');
yimg=ximg;

%DWT
[xar,xhr,xvr,xdr]=dwt2(ximg,'Haar');

%figure, imshow(xar);title('1st level DWT Image');

%2ND_DWT
[xaar,xhhr,xvvr,xddr]=dwt2(xar,'Haar'); 
%figure, imshow(xaar);title('2nd level DWT Image');

%dct
blocksize=8;
i = 0;
for j = 0: blocksize - 1
DCT_trans(i + 1, j + 1) = sqrt(1 / blocksize) ...
* cos ((2 * j + 1) * i * pi / (2 * blocksize));
end
for i = 1: blocksize - 1
for j = 0: blocksize - 1
DCT_trans(i + 1, j + 1) = sqrt(2 / blocksize) ...
* cos ((2 * j + 1) * i * pi / (2 * blocksize));
end
end
sz = size(xaar);
rows = sz(1,1); % finds image's rows and columns
cols = sz(1,2);
%fprintf(1, '\nSize of Data...\n');
%disp(data);
%data=data-128;
%figure;
%imshow(data);
DCT_quantizer = ... % levels for quantizing the DCT block (8x8 matrix)
[ 16 11 10 16 24 40 51 61; ...
12 12 14 19 26 58 60 55; ...
14 13 16 24 40 57 69 56; ...
14 17 22 29 51 87 80 62; ...
18 22 37 56 68 109 103 77; ...
24 35 55 64 81 104 113 92; ...
49 64 78 87 103 121 120 101; ...
72 92 95 98 112 100 103 99 ];
quant_multiple = 0.05; % set the multiplier to change size of quant. levels
% (The values of this defines the number of zero
% coefficients)
% Take DCT of blocks of size blocksize
fprintf(1, '\nFinding the DCT and quantizing...\n');
%starttime = cputime; % "cputime" is an internal cpu time counter
%jpeg_img = data - data; % zero the matrix for the compressed image
DCT_matrix=double(zeros(8,8));
%fprintf(1, '\nSize of Data...\n');
%disp(data);
%fprintf(1, '\nSize of DCT_matrix...\n');
%disp(DCT_matrix);
%fprintf(1, '\nSize of DCT_trans...\n');
%disp(DCT_trans);
for row = 1: blocksize: rows
for col = 1: blocksize: cols
% take a block of the image:
%DCT_matrix = data(row: row + blocksize-1, col: col + blocksize-1);
DCT_matrix(1:8,1:8) = xaar(row:row + blocksize-1,col:col + blocksize-1);
% perform the transform operation on the 2-D block
%fprintf(1, '\n row = %d    col = %d      Size of DCT_matrix...\n',row,col);
%disp(DCT_matrix);


%DCT_matrix(1:1,8:8) =DCT_trans *double( DCT_matrix)*DCT_trans';
DCT_matrix(1:8,1:8) =DCT_trans* double(DCT_matrix) *DCT_trans';
% quantize it (levels stored in DCT_quantizer matrix):
%DCT_matrix = floor (DCT_matrix ...
%./ (DCT_quantizer(1:blocksize, 1:blocksize) * quant_multiple)+0.5);
% place it into the compressed-image matrix:
jpeg_img(row:row + blocksize-1,col:col + blocksize-1) = DCT_matrix(1:8,1:8);
end
end
%figure;
%imshow(jpeg_img);title('DCT image');
%fprintf(1, 'Reconstructing quantized values and taking the inverse DCT...\n');
%svd
[ur, sr, vr]=svd(jpeg_img);
%figure,imshow(sr);title('svd in original Image');
   
   
%watermar_dct
wi=(imread('knee.jpg'));
wr=imresize(wi,[128 128]);
w=rgb2gray(wr);
figure,imshow(w);title('watermark Image');
 
 %watermark_dct
 blocksize=8;
i = 0;
for j = 0: blocksize - 1
DCT_trans(i + 1, j + 1) = sqrt(1 / blocksize) ...
* cos ((2 * j + 1) * i * pi / (2 * blocksize));
end
for i = 1: blocksize - 1
for j = 0: blocksize - 1
DCT_trans(i + 1, j + 1) = sqrt(2 / blocksize) ...
* cos ((2 * j + 1) * i * pi / (2 * blocksize));
end
end
sz = size(w);
rows = sz(1,1); % finds image's rows and columns
cols = sz(1,2);
%fprintf(1, '\nSize of Data...\n');
%disp(data);
%data=data-128;
%figure;
%imshow(data);
DCT_quantizer = ... % levels for quantizing the DCT block (8x8 matrix)
[ 16 11 10 16 24 40 51 61; ...
12 12 14 19 26 58 60 55; ...
14 13 16 24 40 57 69 56; ...
14 17 22 29 51 87 80 62; ...
18 22 37 56 68 109 103 77; ...
24 35 55 64 81 104 113 92; ...
49 64 78 87 103 121 120 101; ...
72 92 95 98 112 100 103 99 ];
quant_multiple = 0.05; % set the multiplier to change size of quant. levels
% (The values of this defines the number of zero
% coefficients)
% Take DCT of blocks of size blocksize
fprintf(1, '\nFinding the DCT and quantizing...\n');
%starttime = cputime; % "cputime" is an internal cpu time counter
%jpeg_img = data - data; % zero the matrix for the compressed image
DCT_matrix=double(zeros(8,8));
%fprintf(1, '\nSize of Data...\n');
%disp(data);
%fprintf(1, '\nSize of DCT_matrix...\n');
%disp(DCT_matrix);
%fprintf(1, '\nSize of DCT_trans...\n');
%disp(DCT_trans);
for row = 1: blocksize: rows
for col = 1: blocksize: cols
% take a block of the image:
%DCT_matrix = data(row: row + blocksize-1, col: col + blocksize-1);
DCT_matrix(1:8,1:8) = w(row:row + blocksize-1,col:col + blocksize-1);
% perform the transform operation on the 2-D block
%fprintf(1, '\n row = %d    col = %d      Size of DCT_matrix...\n',row,col);
%disp(DCT_matrix);


%DCT_matrix(1:1,8:8) =DCT_trans *double( DCT_matrix)*DCT_trans';
DCT_matrix(1:8,1:8) =DCT_trans* double(DCT_matrix) *DCT_trans';
% quantize it (levels stored in DCT_quantizer matrix):
%DCT_matrix = floor (DCT_matrix ...
%./ (DCT_quantizer(1:blocksize, 1:blocksize) * quant_multiple)+0.5);
% place it into the compressed-image matrix:
wjpeg_img(row:row + blocksize-1,col:col + blocksize-1) = DCT_matrix(1:8,1:8);
end
end
%figure;
%imshow(wjpeg_img);title('DCT image');
%fprintf(1, 'Reconstructing quantized values and taking the inverse DCT...\n');

%watermar_svd
[u2r, s2r, v2r]=svd(wjpeg_img);
%figure,imshow(s2r);title('SVD in watermar Image');
    
%---------------------------------------------------------------------------

%embedding
alpha=.50;
 a=alpha*s2r;
 S2=sr+a;
  %figure, imshow(S2);title('embedding Image');
  
  
%Inverse_svd

IS=ur*S2*vr';
%figure,imshow(IS);title('Inverse_svd Image');

%inverse_dct
for row = 1: blocksize: rows
for col = 1: blocksize: cols
% take a block of the image:
IDCT_matrix(1:8,1:8) = IS(row:row + blocksize-1,col:col + blocksize-1);
%IDCT_matrix = double(IDCT_matrix) ...
%.* (DCT_quantizer(1:blocksize, 1:blocksize) * quant_multiple);
% perform the inverse DCT:
IDCT_matrix(1:8,1:8) = DCT_trans' * double(IDCT_matrix) * DCT_trans;
% place it into the reconstructed image:
recon_img(row:row + blocksize-1,col:col + blocksize-1) = IDCT_matrix(1:8,1:8);
end
end
%data=data+128;
%recon_img=recon_img+128;
%fprintf(1, '\nSize of recon_img...\n');
disp(size(recon_img));
%image=zeros(size(data));
%figure;
%image=uint8(recon_img);
%imshow(image);title('Inverse DCT Image');
%imshow(recon_img);


%inverse_2dwt
RI2Dw=idwt2(recon_img,xhhr,xvvr,xddr,'haar');
%figure, imshow(RI2Dw);title('inverse_2nd level DWT Image');

%inverse_dwt
RIDw=idwt2(RI2Dw,xhr,xvr,xdr,'haar');
%figure, imshow(RIDw);title('inverse_1st level DWT Image');
I2= uint8(RIDw);
%ZP=cat(3,uint8(RIDw),g,b);

%imwrite(uint8(ZP),'C:\Users\Swarnali Sadhukhan\Desktop\swarnali\huffman\New folder\attack\Encrypted.jpg');

%converting an image to grayscale
%I2=rgb2gray(ZP); 
figure, imshow(I2);title('watermarked image');
%I= im2bw(I2);

%size of the image
[m,n]=size(I2);
Totalcount=m*n;

%pro array is having the probabilities
sym_prob= zeros(1,256);

 for l = 0:255
    for i = 1:n  %Looping through the column.
         for j = 1:m     
            if I2(i,j) == l 
                     sym_prob(l+1) = sym_prob(l+1) + 1;
            end
        end
    end
end

for i = 1:256
        Probimg(i)=sym_prob(i)/(512*512);
end
sum=0;
for i = 1:256
     sum= sum +Probimg(i);
end


%Symbols for an image
symbols = 0:255;

%function which converts array to vector
vec_size = 1;
for p = 1:m
for q = 1:n
newvec(vec_size) = I2(p,q);
vec_size = vec_size+1;
end
end

%Huffman code Dictionary and encoading
[dict,avglen]=huffmandict(symbols,Probimg);
%dict = huffmandict(symbols,Probimg);
comp= huffmanenco(newvec,dict);
%figure, imshow(comp);title('Main embedded image');
%convertign dhsig1 double to dhsig uint8
edhsig = uint8(comp);

%vector to array conversion

edec_row=sqrt(length(comp));
edec_col=edec_row;

%variables using to convert vector 2 array
earr_row = 1;
earr_col = 1;
evec_si = 1;

for x = 1:m
for y = 1:n
eback(x,y)=comp(evec_si);
earr_col = earr_col+1;
evec_si = evec_si + 1;
end
earr_row = earr_row+1;
end
figure,imshow(eback); title('encoded watermarked  Image');
%----------------------------------------------------------------------------
%writeimg=imwrite(uint8(Z),'C:\Users\Swarnali Sadhukhan\Desktop\my matlab\single-multilayer\embedded.jpg');


%readimg=imread('C:\Users\Swarnali Sadhukhan\Desktop\my matlab\single-multilayer\embedded.jpg');
%figure, imshow(readimg);title('embedded image');
%imwrite(uint8(ZP),'C:\Users\Sajal Chowdhury\Desktop\swarnali\huffman\New folder\attack\Watermarked.jpg');
 



%----------------------------------------------------------------------------

%Huffman Decoding
dhsig1 = huffmandeco(comp,dict);
%convertign dhsig1 double to dhsig uint8
dhsig = uint8(dhsig1);

%vector to array conversion
dec_row=sqrt(length(dhsig));
dec_col=dec_row;

%variables using to convert vector 2 array
arr_row = 1;
arr_col = 1;
vec_si = 1;

for x = 1:m
for y = 1:n
back(x,y)=dhsig(vec_si);
arr_col = arr_col+1;
vec_si = vec_si + 1;
end
arr_row = arr_row+1;
end


%converting image from grayscale to rgb
[deco, map] = gray2ind(back,256);
%deco = gray2ind(back,256);
%RGB = ind2rgb(deco,map);
%figure,imshow(deco); title('decoded.JPG');



%DWT
[zar,zhr,zvr,zdr]=dwt2(deco,'Haar');
%[zar,zhr,zvr,zdr]=dwt2(RIDw,'Haar');

%figure, imshow(zar);title('1st level DWT on embedded Image');

%2ND_DWT
[zaar,zhhr,zvvr,zddr]=dwt2(zar,'Haar'); 
%figure, imshow(zaar);title('2nd level DWT  on embedded Image');

 %dct

blocksize=8;
i = 0;
for j = 0: blocksize - 1
DCT_trans(i + 1, j + 1) = sqrt(1 / blocksize) ...
* cos ((2 * j + 1) * i * pi / (2 * blocksize));
end
for i = 1: blocksize - 1
for j = 0: blocksize - 1
DCT_trans(i + 1, j + 1) = sqrt(2 / blocksize) ...
* cos ((2 * j + 1) * i * pi / (2 * blocksize));
end
end
sz = size(zaar);
rows = sz(1,1); % finds image's rows and columns
cols = sz(1,2);
%fprintf(1, '\nSize of Data...\n');
%disp(data);
%data=data-128;
%figure;
%imshow(data);
DCT_quantizer = ... % levels for quantizing the DCT block (8x8 matrix)
[ 16 11 10 16 24 40 51 61; ...
12 12 14 19 26 58 60 55; ...
14 13 16 24 40 57 69 56; ...
14 17 22 29 51 87 80 62; ...
18 22 37 56 68 109 103 77; ...
24 35 55 64 81 104 113 92; ...
49 64 78 87 103 121 120 101; ...
72 92 95 98 112 100 103 99 ];
quant_multiple = 0.05; % set the multiplier to change size of quant. levels
% (The values of this defines the number of zero
% coefficients)
% Take DCT of blocks of size blocksize
fprintf(1, '\nFinding the DCT and quantizing...\n');
%starttime = cputime; % "cputime" is an internal cpu time counter
%jpeg_img = data - data; % zero the matrix for the compressed image
DCT_matrix=double(zeros(8,8));
%fprintf(1, '\nSize of Data...\n');
%disp(data);
%fprintf(1, '\nSize of DCT_matrix...\n');
%disp(DCT_matrix);
%fprintf(1, '\nSize of DCT_trans...\n');
%disp(DCT_trans);
for row = 1: blocksize: rows
for col = 1: blocksize: cols
% take a block of the image:
%DCT_matrix = data(row: row + blocksize-1, col: col + blocksize-1);
DCT_matrix(1:8,1:8) = zaar(row:row + blocksize-1,col:col + blocksize-1);
% perform the transform operation on the 2-D block
%fprintf(1, '\n row = %d    col = %d      Size of DCT_matrix...\n',row,col);
%disp(DCT_matrix);


%DCT_matrix(1:1,8:8) =DCT_trans *double( DCT_matrix)*DCT_trans';
DCT_matrix(1:8,1:8) =DCT_trans* double(DCT_matrix) *DCT_trans';
% quantize it (levels stored in DCT_quantizer matrix):
%DCT_matrix = floor (DCT_matrix ...
%./ (DCT_quantizer(1:blocksize, 1:blocksize) * quant_multiple)+0.5);
% place it into the compressed-image matrix:
ewjpeg_img(row:row + blocksize-1,col:col + blocksize-1) = DCT_matrix(1:8,1:8);
end
end


%svd
[uz, sz, vz]=svd(ewjpeg_img);
%figure,imshow(sz);title('svd in on embedded Image');

%watermar extraction
SZ=(sz-sr)/alpha;%for watermark image extraction
%Inverse_svd
Isz=u2r*SZ*v2r';
%figure,imshow(Isz);title('Inverse_svd Image');

%inverse_dct
for row = 1: blocksize: rows
for col = 1: blocksize: cols
% take a block of the image:
IDCT_matrix(1:8,1:8) = Isz(row:row + blocksize-1,col:col + blocksize-1);
%IDCT_matrix = double(IDCT_matrix) ...
%.* (DCT_quantizer(1:blocksize, 1:blocksize) * quant_multiple);
% perform the inverse DCT:
IDCT_matrix(1:8,1:8) = DCT_trans' * double(IDCT_matrix) * DCT_trans;
% place it into the reconstructed image:
recon_img(row:row + blocksize-1,col:col + blocksize-1) = IDCT_matrix(1:8,1:8);
end
end
%data=data+128;
%recon_img=recon_img+128;
fprintf(1, '\nSize of recon_img...\n');
disp(size(recon_img));
%image=zeros(size(data));

image=uint8(recon_img);
%imshow(image);title('Inverse DCT Image');
%imshow(recon_img);


%ew=cat(3,uint8(ewimage),wg,wb);
figure, imshow(image);title('extracted watermark image');



  [peaksnr, snr] = psnr(ZP, ximg);
fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
fprintf('\n The SNR value is %0.4f \n', snr);
  
[ssimval, ssimmap] = ssim(image,w);
fprintf('The SSIM value is %0.4f.\n',ssimval);

figure;imhist(ximg);title('Histogram of host image')
figure;imhist(I2);title('Histogram of watermarked image')
figure;imhist(comp);title('Histogram of  image')
  
