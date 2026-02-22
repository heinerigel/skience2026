from obspy.clients.fdsn import Client
from obspy import *
import os
import datetime
#cl = Client("http://tarzan")
import glob
# All files and directories ending with .txt and that don't begin with a dot:

t1 = UTCDateTime("2025-03-19T00:00")
t2 = UTCDateTime("2025-03-23T00:00")

stations = ["A3P1","A3P2","A3P3","A3P4","A3P5","A3P6","A3P7","A3P8"]

for stat in stations:
    tjul = int(t1.julday)
    ejul = int(t2.julday)
    dat = t1
    while tjul <= ejul:
        print(dat," ",tjul)
        try:
            st = read("./data/%04d%03d_%s_*.m"%(dat.year,tjul,stat))
            st.resample(sampling_rate=200,no_filter=False)

            new_traces = Stream()
            have_data = False
            for tr in st:
                new_traces.append(tr)
            st2 = Stream(new_traces)

            print(st2)
        #myday = int(st[0].stats.starttime.julday)

        # open catalog file in read and write mode in case we are continuing d/l,
        # so we can append to the file
            for tr in st2:
                pathyear = str(tr.stats.starttime.year)
                mydatapath = os.path.join("./data_sds/", pathyear)
            # create datapath 
                if not os.path.exists(mydatapath):
                    os.mkdir(mydatapath)

                mydatapath = os.path.join(mydatapath, tr.stats.network)

                if not os.path.exists(mydatapath):
                    os.mkdir(mydatapath)


                mydatapath = os.path.join(mydatapath, tr.stats.station)
            # create datapath 
                if not os.path.exists(mydatapath):
                   os.mkdir(mydatapath)

                mydatapathchannel = os.path.join(mydatapath,tr.stats.channel + ".D")

                if not os.path.exists(mydatapathchannel):
                    os.mkdir(mydatapathchannel)

                netFile =\
                "%s.%s.%s.%s.D.%s.%03d"%(tr.stats.network,tr.stats.station,tr.stats.location,tr.stats.channel,pathyear,tr.stats.starttime.julday)
                netFileout = os.path.join(mydatapathchannel, netFile)

                # try to open File
                try:
                    netFileout = open(netFileout, 'ab')
                except:
                    netFileout = open(netFileout, 'w')

                # header of the stream object which contains the output of the ADR
                tr.write(netFileout,format='MSEED')
                netFileout.close()

            dat += 24*3600
            tjul = int(dat.julday)

        except:
            dat += 24*3600
            tjul = int(dat.julday)
            continue;
                                             
