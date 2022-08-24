#Preprocessing script to clean dataset and develop the image files allowing loading of CNN models to explore performance of regular Argmax classification

# In [5]
df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)

# In [8]
df = df[df['x28']==96]

df= df[['time','y', 'x2','x3', 'x5', 'x6','x8', 'x10', 'x13', 'x15',  'x21', 'x22', 'x26', 'x27', 'x30',
       'x34', 'x39', 'x42', 'x48',  'x50', 'x52', 'x54', 'x55', 'x56', 'x58',  'x60']]

# In [12]
df['y'][df['time']==pd.to_datetime('1999-05-13 18:14:00')]=0
df['y'][df['time']==pd.to_datetime('1999-05-14 09:30:00')]=0

#In [14]
df = df[df['time'] != pd.to_datetime('1999-05-12 22:48:00')]
print('Length of df is ',len(df))

# In [15]
dfy = pd.DataFrame(columns = ['y', 'gaps'])
dfy['y'] = df['y'][df['y']==1]

Inds = dfy.index
gaps=[]
for i in range(len(Inds)):
    if i != 0:
        gaps.append(Inds[i] - Inds[i-1])
    else:
        gaps.append(Inds[i])
dfy['gaps'] = gaps
print('Mininmum samples between faults: ', min(dfy['gaps']))

dfy.head()

# In [17]
Ind_6 = dfy[dfy['gaps']<7].index
for i in Ind_6:
    for j in range( dfy.loc[i][1] ):
        df['y'].loc[i-j] = 10

df = df[df['y']!=10]

print('Number of remaining faults is: ', df['y'].sum())
print('remaing shape of df is: ', df.shape) 

# In [18 - 20]
df=df.reset_index()

master_index = df['time']

dfy2 = pd.DataFrame(columns = ['y', 'gaps'])
dfy2['y'] = df['y'][df['y']==1]

Inds2 = dfy2.index
gaps=[]
for i in range(len(Inds2)):
    if i != 0:
        gaps.append(Inds2[i] - Inds2[i-1])
    else:
        gaps.append(Inds2[i])
dfy2['gaps'] = gaps
print('Mininmum samples between faults: ', min(dfy2['gaps']), 'Maximum samples between faults is: ', max(dfy2['gaps']) )

dfy2.head()

# In [24]
vel_cols = ['time', 'y']
acc_cols = ['time', 'y']
for item in df.columns[3:]:
    vel_cols.append(item + '_vel')
    acc_cols.append(item + '_acc')

# In[25]
df_vel = pd.DataFrame(columns = vel_cols)
df_vel['time'] = df['time']
df_vel['y'] = df['y']

df_vel.iloc[0,2:] = 0
df_vel.iloc[1:,2:] = (np.array( df.iloc[1:,3:]) - np.array( df.iloc[0:-1,3:]) )/2

#In[27]
Indx = 0

for row in range(len(dfy2)):
    gap = dfy2.iloc[row,1]
    Indx = Indx + gap 
    df_vel.iloc[Indx+1, 2:] = 0

# In[28]
df_acc = pd.DataFrame(columns = acc_cols)
df_acc['time'] = df['time']
df_acc['y'] = df['y']

df_acc.iloc[0,2:] = 0
df_acc.iloc[1:,2:] = (np.array( df_vel.iloc[1:,2:]) - np.array( df_vel.iloc[0:-1,2:]) )/2

# In[29]
Indx = 0

for row in range(len(dfy2)):
    gap = dfy2.iloc[row,1]
    Indx = Indx + gap 
    df_acc.iloc[Indx+1, 2:] = 0

# In[31]
image_count = len(df) - 5 - sum(dfy2['gaps'][dfy2['gaps']<=6]) - 6*(sum(df['y']) - len(dfy2[dfy2['gaps']<=6]) )

# In[32]
df_s = df.drop(['index','time', 'y'], axis = 1)
df_vel_s = df_vel.drop(['time', 'y'], axis = 1)
df_acc_s = df_acc.drop(['time', 'y'], axis = 1)

#y_true = df[['y', 'time']].values
y_true = df['y']
print('Check, sum of y_true:', sum(y_true) )

# In[33]
scaler = MinMaxScaler(feature_range=(0,1) )

df_s=scaler.fit_transform(df_s)
df_vel_s=scaler.fit_transform(df_vel_s)
df_acc_s=scaler.fit_transform(df_acc_s)

# In[34]
Ws, Wv, Wa = df_s.shape[1], df_vel_s.shape[1], df_acc_s.shape[1]

# In[36]
print('Build an image array with 6 measurement, velocity, & acceleration time slices on each 12 minute time range image.')
images = np.zeros((image_count,18,Ws))
Y = np.zeros((image_count)).astype(int)
lbl = np.zeros((image_count)).astype(int)
img_index = np.zeros((image_count)).astype(pd.Timestamp)

# In[38]
print('image#, first row, last row,         y')
i = 0
for c in range(image_count):
       
    images[c,0:6,:] = df_s[i:i+6,:]         # Load 6 rows of position values
    
    images[c,6:12,0:] = df_vel_s[i:i+6,:]   # Load 6 rows of 'velocity / first derivative' values
    
    images[c,12:18,0:] = df_acc_s[i:i+6,:]   # Load 6 rows of 'acceleration / second derivative' values
    
    lbl[c] = y_true[i+5]                    # label images with faults as 1
    img_index[c] = master_index[i+5]
    
    Y[c] = 1*(y_true[i+10]==1) + 1*(y_true[i+9]==1) + 1*(y_true[i+8]==1) + 1*(y_true[i+7]==1) + 1*(y_true[i+6]==1) + 2*(y_true[i+5]==1)

    if Y[c] == 2: 
        print(('{:^10}{:^14}{:^10}{:^10}').format(c, i, i+5, y_true[i+5])) 
        i = i + 6
    else:
        i = i + 1

print(('{:^10}{:^14}{:^10}{:^10}').format(c, i-1, i+5-1, y_true[i+5-1])) 

# In[41]
two, one, zero, tot = (Y==2).sum(), (Y==1).sum(), (Y==0).sum(), len(Y)

# In[42]
print('12 images leading up to a fault and the first image after a reset are shown below. The incremental changes between each image is shown next.')
print('/nImage and prediction for Image 244 through 255')
plt.figure(figsize=(15,8))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i+244], cmap='gray')
    plt.xlabel(str(lbl[i+244]) + str(Y[i+244])) #(class_labels[y_pred[i+j]]+' '+str(y_pred[i+j]==y_test[i+j]))
plt.show()

# In[49 - 54]
# Build train and test set from the image dataset also splitting the image time index.
x_train, x_test, y_train, y_test, train_indx, test_indx = train_test_split(images, Y, img_index, stratify = Y, test_size=0.35, random_state = 0)

tr_s, te_s = x_train.shape[0], x_test.shape[0]

# Save a viewable copy of x_train & x_test to display results
x_te = x_test
x_tr = x_train

x_train = x_train.reshape(tr_s,18,24,1)
x_test = x_test.reshape(te_s,18,24,1)

x_train = x_train - 0.5
x_test = x_test - 0.5

# In[55]
print('Define Adjusted_Pred')
def Adjusted_Pred(y_prob, threshold):
    A_prob = np.zeros(y_prob.shape).astype(float)
    A_prob[:,0:3] = y_prob
    
    A_prob[:,1] = A_prob[:,1]*(A_prob[:,1]>=(1-threshold)*5/6)  # Sets prob to 0 if it is less than average frequency of 1
    A_prob[:,2] = A_prob[:,2]*(A_prob[:,2]>=(1-threshold)/6)  # Sets prob to 0 if it is less than average frequency of 2
    for j in range ( len(y_prob) ):
        if sum(A_prob[j,1:3])>0:
            A_prob[j,0] = A_prob[j,0]*(A_prob[j,0]>=threshold)  # normal probability only if probability exceeds the ratio of normals in the set  
    A_pred = A_prob[:,0]
    A_pred = A_prob[:,0:3].argmax(axis=1) 
    return A_pred

# In[56]
print('Define Train_Test_Time_Plot')
def Train_Test_Time_Plot(tracks, actual_tr, pred_tr, prob_tr, Indx_tr, actual_te, pred_te, prob_te, Indx_te, mstr_indx, mkr_size, a, b, title1, fig_height=6, show_normal=True):
    #Rge=range(a,b)
    if tracks == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,fig_height))
    else:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12,fig_height))
        
    ax1.plot(Indx_tr[:], pred_tr[:], label = "Train Prediction", c ='b', ls = 'none', marker = 'x', markersize = mkr_size)
    ax1.plot(Indx_te[:], pred_te[:], label = "Test Prediction", c = 'b', ls = 'none', marker = 'o', markersize = mkr_size)
    ax1.plot(Indx_tr[actual_tr==2], actual_tr[actual_tr==2], label = "Actual Fault",  ls = 'none', marker = 'D', markerfacecolor='none', markeredgecolor='black', markersize = 2*mkr_size)
    ax1.plot(Indx_te[actual_te==2], actual_te[actual_te==2], ls = 'none', marker = 'D', markerfacecolor='none', markeredgecolor='black', markersize = 2*mkr_size)
    ax1.plot(Indx_tr[actual_tr==1], actual_tr[actual_tr==1], label = "Actual Warning", ls = 'none', marker = 'o', markerfacecolor='none', markeredgecolor='gray', markersize = 2*mkr_size)
    ax1.plot(Indx_te[actual_te==1], actual_te[actual_te==1], ls = 'none', marker = 'o', markerfacecolor='none', markeredgecolor='gray', markersize = 2*mkr_size)
    ax1.set_xlim(mstr_indx[a],mstr_indx[b])
    ax1.legend()
    if tracks == 3:
        ax2.plot(Indx_tr[:], actual_tr[:], label ="Actual Label", c ='black', ls = 'none', marker = 'x', markersize = mkr_size)
        ax2.plot(Indx_te[:], actual_te[:], label ="Actual Label", c ='red', ls = 'none', marker = 'D', markersize = mkr_size)
        ax2.set_xlim(mstr_indx[a],mstr_indx[b])
        ax2.set_title('Actual Label, 2 = Fault, 1 = Warning')
    
    if show_normal == True:
        ax3.plot(Indx_tr[:], prob_tr[:,0], label ="Train Normal", c='g', ls = 'none', marker = 'x', markersize = mkr_size) 
        ax3.plot(Indx_te[:], prob_te[:,0], label ="Test Normal", c='g', ls = 'none', marker = 'o', markersize = mkr_size) 
        
    ax3.plot(Indx_tr[:], prob_tr[:,1], label ="Train Warning", c='gold', ls = 'none', marker = 'x', markersize = mkr_size)
    ax3.plot(Indx_te[:], prob_te[:,1], label ="Test Warning", c='gold', ls = 'none', marker = 'o', markersize = mkr_size)
    
    ax3.plot(Indx_tr[:], prob_tr[:,2], label ="Train Fault", c = 'r', ls = 'none', marker = 'x', markersize = mkr_size)
    ax3.plot(Indx_te[:], prob_te[:,2], label ="Test Fault", c = 'r', ls = 'none', marker = 'o', markersize = mkr_size)
    ax3.set_xlim(mstr_indx[a],mstr_indx[b])
    ax3.legend()
    ax1.set_title(('{}{}{}').format('Predicted Impending Fault on ', title1, '\n  Prediction'))
    
    ax3.set_title('Probability of Class')
    plt.xlabel('Day - Time')
    #plt.legend()
    plt.tight_layout()
    plt.show()

# In[60]
#  Set threshold for the normal class
TH_N = 1-(one+two)/6168   # 1 minus the average frequency for labels 1 through 2


