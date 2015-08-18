%**************************************************************************
%函数名称：DAG_classifier()
%参数：sample：待识别样本特征
%返回值：y：分类结果
%函数功能：利用有向无环图支持向量机对测试样本分类
%**************************************************************************
function y=DAG_classifier(sample,pos)
load svmStruct svmStruct;
a=1;
b=10;
for i=1:9
    %支持向量机两类分类
    G=svmclassify(svmStruct(a,b),sample);
    if(G==-1)
        b=b-1;
    else
        a=a+1;
    end
end

if(G==-1)
    y=pos(a)-1;
else
    y=pos(b)-1;
end



