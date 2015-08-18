%**************************************************************************
%                         有向无环图支持向量机分类器分析analysis.m
%**************************************************************************
clear all;close all;
%手写数字0的训练样本特征获取*************************************************
load mnist_all train0;
[r,c]=size(train0);
pattern(1).num=r;
for i=1:r
    I=train0(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(1).feature(1:25,i)=feature;
end
 
%手写数字1的训练样本特征获取*************************************************
load mnist_all train1;
[r,c]=size(train1);
pattern(2).num=r;
for i=1:r
    I=train1(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(2).feature(1:25,i)=feature;
end

%手写数字2的训练样本特征获取*************************************************
load mnist_all train2;
[r,c]=size(train2);
pattern(3).num=r;
for i=1:r
    I=train2(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(3).feature(1:25,i)=feature;
end

%手写数字3的训练样本特征获取*************************************************
load mnist_all train3;
[r,c]=size(train3);
pattern(4).num=r;
for i=1:r
    I=train3(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(4).feature(1:25,i)=feature;
end

%手写数字4的训练样本特征获取*************************************************
load mnist_all train4;
[r,c]=size(train4);
pattern(5).num=r;
for i=1:r
    I=train4(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(5).feature(1:25,i)=feature;
end

%手写数字5的训练样本特征获取*************************************************
load mnist_all train5;
[r,c]=size(train5);
pattern(6).num=r;
for i=1:r
    I=train5(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(6).feature(1:25,i)=feature;
end

%手写数字6的训练样本特征获取*************************************************
load mnist_all train6;
[r,c]=size(train6);
pattern(7).num=r;
for i=1:r
    I=train6(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(7).feature(1:25,i)=feature;
end

%手写数字7的训练样本特征获取*************************************************
load mnist_all train7;
[r,c]=size(train7);
pattern(8).num=r;
for i=1:r
    I=train7(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(8).feature(1:25,i)=feature;
end

%手写数字8的训练样本特征获取*************************************************
load mnist_all train8;
[r,c]=size(train8);
pattern(9).num=r;
for i=1:r
    I=train8(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(9).feature(1:25,i)=feature;
end 

%手写数字9的训练样本特征获取*************************************************
load mnist_all train9;
[r,c]=size(train9);
pattern(10).num=r;
for i=1:r
    I=train9(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    feature=feature_extract2(A); 
    pattern(10).feature(1:25,i)=feature;
end 
%**************************************************************************

%**************************************************************************
save templet pattern                                %建立样本特征库
%**************************************************************************

%PCA***********************************************************************
for i=1:10
    Feature=[Feature;pattern(i).feature'];
end
[pc,score,latent,tsquare]=princomp(Feature);
%训练svmStruct*************************************************************
pos=DAG_train();
%**************************************************************************

% toc;
%利用有向无环图支持向量机分类器进行错误率分析**********************************
pc=zeros(1,10);
digital=0:1:9;
load mnist_all test0;
[r,c]=size(test0);  
sum=0;
for i=1:r
    I=test0(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);  
    if y~=0
        sum=sum+1;
    end
end
pc(1)=(r-sum)/r;

load mnist_all test1;
[r,c]=size(test1);
sum=0;
for i=1:r
    I=test1(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);   
    if y~=1
        sum=sum+1;
    end
end
pc(2)=(r-sum)/r;

load mnist_all test2;
[r,c]=size(test2);
sum=0;
for i=1:r
    I=test2(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);    
    if y~=2
        sum=sum+1;
    end
end
pc(3)=(r-sum)/r;

load mnist_all test3;
[r,c]=size(test3);
sum=0;
for i=1:r
    I=test3(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);    
    if y~=3
        sum=sum+1;
    end
end
pc(4)=(r-sum)/r;

load mnist_all test4;
[r,c]=size(test4); 
sum=0;
for i=1:r
    I=test4(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);  
    if y~=4
        sum=sum+1;
    end
end
pc(5)=(r-sum)/r;

load mnist_all test5;
[r,c]=size(test5);
sum=0;
for i=1:r
    I=test5(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);    
    if y~=5
        sum=sum+1;
    end
end
pc(6)=(r-sum)/r;

load mnist_all test6;
[r,c]=size(test6);
sum=0;
for i=1:r
    I=test6(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);    
    if y~=6
        sum=sum+1;
    end
end
pc(7)=(r-sum)/r;

load mnist_all test7;
[r,c]=size(test7);
sum=0;
for i=1:r
    I=test7(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);   
    if y~=7
        sum=sum+1;
    end
end
pc(8)=(r-sum)/r;

load mnist_all test8;
[r,c]=size(test8);
sum=0;
for i=1:r
    I=test8(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos);
    if y~=8
        sum=sum+1;
    end
end
pc(9)=(r-sum)/r;

load mnist_all test9;
[r,c]=size(test9);
sum=0;
for i=1:r
    I=test9(i,:);
    A=reshape(I,sqrt(c),sqrt(c));
    sample=feature_extract2(A); 
    y=DAG_classifier(sample,pos); 
    if y~=9
        sum=sum+1;
    end
end
pc(10)=(r-sum)/r;

figure
xlabel('手写数字');ylabel('正确识别率');
semilogy(digital,pc,'b*-');
%**************************************************************************
% disp(['样本特征库建立时间: ', num2str(toc),'s'])