%**************************************************************************
%�������ƣ�DAG_classifier()
%������sample����ʶ����������
%����ֵ��y��������
%�������ܣ����������޻�ͼ֧���������Բ�����������
%**************************************************************************
function y=DAG_classifier(sample,pos)
load svmStruct svmStruct;
a=1;
b=10;
for i=1:9
    %֧���������������
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



