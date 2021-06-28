import pandas as pd
import numpy as np
n_sample=100000
df1=pd.read_csv("BOUN_TCP_Anon.csv",nrows=n_sample)
df2=pd.read_csv("BOUN_UDP_Anon.csv",nrows=n_sample)
df=pd.concat([df1,df2],axis=0,ignore_index=True)
print(df.head())
df['TCP_Protocol'].value_counts()
df['Source_ip'].value_counts()
l=['10.50.197.71','10.50.192.199']
print(df)
for i in range(0,len(df)):
    if(df['Source_ip'][i] not in l):
        df['Source_ip'][i]='other_src_ip'
src_dummies=pd.get_dummies(df['Source_ip'])
src_dummies=src_dummies.rename(columns={"10.50.197.71":"src_71","10.50.192.199":"src_199"})
print(src_dummies.head())
df_new=src_dummies
df['Destination_IP'].value_counts()
l1=['10.50.197.71','10.50.192.199']
for i in range(0,len(df)):
    if(df['Destination_IP'][i] not in l1):
        df['Destination_IP'][i]='other_dest_ip'
dst_dummies=pd.get_dummies(df['Destination_IP'])
dst_dummies=dst_dummies.rename(columns={"10.50.197.71":"dst_71","10.50.192.199":"dst_199"})
df_new=pd.concat([df_new,dst_dummies],axis=1)
for i in range(0,len(df)):
    if(pd.isnull(df['Source_Port'][i])):
        df['Source_Port'][i]=-1
df['Source_Port'].value_counts()
for i in range(0,len(df)):
    if(pd.isnull(df['Destination_Port'][i])):
        df['Destination_Port'][i]=-1
df['Destination_Port'].value_counts()
for i in range(0,len(df)):
    if(pd.isnull(df['SYN'][i])):
        df['SYN'][i]="NaN"
for i in range(0,len(df)):
    if(pd.isnull(df['ACK'][i])):
        df['ACK'][i]="NaN"
for i in range(0,len(df)):
    if(pd.isnull(df['RST'][i])):
        df['RST'][i]="NaN"
print(df['SYN'].value_counts())
print(df['ACK'].value_counts())
print(df['RST'].value_counts())
for i in range(0,len(df)):
    if(df['Source_Port'][i]==-1):
        df['Source_Port'][i]='srcport_null'
    elif(df['Source_Port'][i]==443):
        df['Source_Port'][i]='srcport_443'
    elif(df['Source_Port'][i]==49543):
        df['Source_Port'][i]='srcport_49543'
    else:
        df['Source_Port'][i]='other_srcport'
    if(df['Destination_Port'][i]==-1):
        df['Destination_Port'][i]='dstport_null'
    elif(df['Destination_Port'][i]==443):
        df['Destination_Port'][i]='dstport_443'
    elif(df['Destination_Port'][i]==49543):
        df['Destination_Port'][i]='dstport_49543'
    else:
        df['Destination_Port'][i]='other_dstport'
srcport_dummies=pd.get_dummies(df['Source_Port'])
dstport_dummies=pd.get_dummies(df['Destination_Port'])
df_new=pd.concat([df_new,srcport_dummies],axis=1)
df_new=pd.concat([df_new,dstport_dummies],axis=1)
for i in range(0,len(df)):
    if(df['SYN'][i]=='Not set'):
            df['SYN'][i]="SYN_NOT_SET"
    else:
        df['SYN'][i]="SYN_Other"
            
    if(df['ACK'][i]=='Set'):
        df['ACK'][i]="ACK_SET"
    else:
        df['ACK'][i]="ACK_Other"
    if(df['RST'][i]=='Not set'):
        df['RST'][i]="RST_NOT_SET"
    else:
        df['RST'][i]="RST_Other"
syn_dummies=pd.get_dummies(df['SYN'])
ack_dummies=pd.get_dummies(df['ACK'])
rst_dummies=pd.get_dummies(df['RST'])
print(syn_dummies.head())
print(ack_dummies.head())
print(rst_dummies.head())
df['TTL'].value_counts()
df_new=pd.concat([df_new,syn_dummies],axis=1)
df_new=pd.concat([df_new,ack_dummies],axis=1)
df_new=pd.concat([df_new,rst_dummies],axis=1)
for i in range(0,len(df)):
    if(df['TTL'][i] != "127"):
        df['TTL'][i]="OTHER_TTL"
    else:
        df['TTL'][i]="TTL_127""
ttl_dummies=pd.get_dummies(df['TTL'])
df_new=pd.concat([df_new,ttl_dummies],axis=1)
df['Frame_length'].value_counts()
for i in range(0,len(df)):
    if(df['Frame_length'][i] == 64):
        df['Frame_length'][i] = "FL_64"
    elif(df['Frame_length'][i] == 2978):
        df['Frame_length'][i] = "FL_2978"
fl_dummies=pd.get_dummies(df['Frame_length'])
df_new=pd.concat([df_new,fl_dummies],axis=1)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
time=df[['Time']]
time=sc.fit_transform(time)
time=pd.DataFrame(time)
time=time.rename(columns={0:"Time"})
df_new=pd.concat([df_new,time],axis=1)
df['TCP_Protocol'].value_counts()
for i in range(0,len(df)):
    if(df['TCP_Protocol'][i]!="TCP"):
        if(df['TCP_Protocol'][i]!="UDP"):
            df['TCP_Protocol'][i]="Others"
print(df['TCP_Protocol'].value_counts())
df_new=pd.concat([df_new,df['TCP_Protocol']],axis=1)
df_new.to_csv("dataset2.csv")
    
    