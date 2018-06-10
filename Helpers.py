

def record(fname):
    
    '''Helper function for option5 that records sound and uses fname
       as the file name.
       @param string saves recorded audio with this name'''
    import pyaudio
    #    import time
    import wave
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 20
    WAVE_OUTPUT_FILENAME = fname
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)
        
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # start_time = time.time()
    # match = id_sample(fname, mode="record")
    # end_time = time.time()

    # print("\n\n===================================")
    # if match[0] == "No match was found":
        # print (match[0])
    # else:
        # print("Your song is:")
        # print(match[0])
        # print("-------")
        # print("Search completed in %g seconds" % (end_time - start_time))
        # print("===================================\n")
        # print("Press 'r' for related songs, or 'Enter' to return to menu")
        # ans1 = raw_input()
        # if ans1 == 'r':
            # print("Related Songs\n=============\n")
            # for name in match[1]:
                # print(name.split("/")[1].strip())
            # print("\nPress any key to return to menu")
            # ans2 = raw_input()
            # menu()
        # else:
            # menu()