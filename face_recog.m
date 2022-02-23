%____ LINEAR ALGEBRA, PROJECT 3
%____ Face Recognition
%____Samin Zare
clear all;
clc;
%____________ STEP 1 _______________
trainpic_num=5;
people_num=40;
pixel_in_pic=112*92;
Train=zeros(pixel_in_pic,people_num*trainpic_num);

for j=1:people_num
    for i=1:trainpic_num
        
        I=imread(['C:\Users\digi max\Desktop\orl_faces/s' int2str(j) '\' int2str(i) '.pgm' ]);
        I=im2double(I);
        %figure, imshow(I);
        
        [r,c]=size(I);
        I=reshape(I,r*c,1); % reshape matrix I as a Vector of 112*92
     
       Train(:,5*(j-1)+i)=I;    % Training Matrix:     Save all I Vectors for each picture((112*92)*200)
        
        
    end
end

%_____________STEP 2_________________
%__To Normalize the Training Matrix__

Mean_Train=mean(Train);
D_Train=zeros(pixel_in_pic,people_num*trainpic_num);
for i= 1: trainpic_num*people_num
    for j=1:pixel_in_pic
    D_Train(j,i)=Train(j,i)-Mean_Train(i); % subtract mean of training vectors from training data to get the normalized training data
    end
end


% %____________ STEP 3_______________
% %___________TEST MATRIX____________


testpic_num=5;
people_num=40;
pixel_in_pic=112*92;
all_pics=people_num*testpic_num;
Test=zeros(pixel_in_pic,all_pics);

for j=1:people_num
    for i=1:testpic_num
        
        I=imread(['C:\Users\digi max\Desktop\orl_faces/s' int2str(j) '\' int2str(i+5) '.pgm' ]);
        I=im2double(I);
        %figure, imshow(I);
        
        [r,c]=size(I);
        I=reshape(I,r*c,1); % reshape matrix I as a Vector of 112*92
     
       Test(:,5*(j-1)+i)=I;    % Test Matrix:     Save all I Vectors for each picture((112*92)*200)
        
     end
end
% To Normalize the Test Matrix
Mean_Test=mean(Test);
D_Test=zeros(pixel_in_pic,all_pics);
for i= 1: all_pics
    for j=1:pixel_in_pic
    D_Test(j,i)=Test(j,i)-Mean_Test(i); % subtract mean of training vectors from training data
    end
end


%___________ STEP 4_________________________
%       ____METHOD 1 _____

d1=zeros(200,1);
right_m1=0;
wrong_m1=0;

for test_vector=1:all_pics

for train_vector=1:all_pics
    
   d1(train_vector)= norm(D_Test(:,test_vector)-D_Train(:,train_vector),1);
        
end

Min_m1=min(d1);
i=find(d1==Min_m1);
 
        if abs(i-test_vector)<=5;
        right_m1=right_m1+1;
        else
            wrong_m1=wrong_m1+1;
        end
    
   
end


right_m1
wrong_m1

%_________ STEP 5 _________

for i=1:200
    
S(i)=svd(D_Train(:,i));
singular_values(i)=diag(S(i));


end


[U,S,V]=svd(D_Train);
Singular_Values=diag(S);

%______ STEP 6_______
% PLot the singluar Values
figure;
plot(Singular_Values);
title('Singluar Values of Normalized Train Matrix')
xlabel('picture number')
ylabel('singular value')
grid on;

Min_SV= min(Singular_Values);
Max_SV=max(Singular_Values);


% Sorted_SV = sort(Singular_Values); % sort singular values to get the 10 biggest ones
% 
% Max_SV= Sorted_SV(:,191:200);
% basic_vectors=zeros(1,10);
% for i=1:10
% basic_vectors(i)= find(singular_values==Max_SV(i));
% end
% %sort(basic_vectors);
% basic_vectors=sort(basic_vectors);
% 
% %[U,S,V]=svd(D_Train);
% %imshow(reshape(U(:,i),r,c),[])
% for i=1:10
%   Q(:,i)=D_Train(:,basic_vectors(i));
% end





%_____________STEP 7 ______________
%_____________Eigen Faces_________
%to see the first 10  basis vectors

for i=1:10
    figure;
    imshow(reshape(U(:,i),r,c),[]);
end


for i=190:200
    figure;
    imshow(reshape(U(:,i),r,c),[]);
end

%___________STEP 8 __________
NK=10;     % number of basis vectors we want to keep
UKEEP=U(:,1:NK);
PROJD=(D_Train'*UKEEP)';



%__________ STEP 9 __________

% To project the normalized test data matrix
PROJT=(D_Test'*UKEEP)';

%_________ STEP 10 __________
%________TEST METHOD 2 ______

d2=zeros(200,1);
right_m2=0;
wrong_m2=0;

for test_vector=1:all_pics

    Test=PROJT(:,test_vector);

for train_vector=1:all_pics
    
   d2(train_vector)= norm(Test-PROJD(:,train_vector),1);
        
end

Min_m2=min(d2);
i=find(d2==Min_m2);
 
        if abs(i-test_vector)<=5;
        right_m2=right_m2+1;
        else
            wrong_m2=wrong_m2+1;
        end
    
   
end


right_m2
wrong_m2

