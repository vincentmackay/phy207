import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import ImageDraw
import PIL
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import random
from matplotlib.patches import Rectangle

def draw(smoothen = True):
    n_points = 10000
    width = 800  # canvas width
    height = 200 # canvas height
    white = (255, 255, 255) # canvas back

    def save():
        master.destroy()

    def paint(event):
        polygon_half_width = 4
        x1, y1 = (event.x - 1), (event.y - 1)
        #canvas.create_oval(x1, y1, x2, y2, fill="black",width=5)
        canvas.create_line(x1+10, y1-5, x1+10, 0, fill="white",width=10)
        canvas.create_line(x1+10, y1+5, x1+10, height, fill="white",width=10)
        draw.line([x1+10, y1-5, x1+10, 0], fill="white",width=10)
        draw.line([x1+10, y1+5, x1+10, height], fill="white",width=10)
        canvas.create_polygon( x1+polygon_half_width, y1+polygon_half_width, x1+polygon_half_width, y1-polygon_half_width, x1-polygon_half_width, y1-polygon_half_width, x1-polygon_half_width, y1+polygon_half_width, fill="black",width=5)
        draw.rectangle( [x1+polygon_half_width, y1+polygon_half_width, x1-polygon_half_width, y1-polygon_half_width], fill="black")


    master = Tk()

    # create a tkinter canvas to draw on
    canvas = Canvas(master, width=width, height=height, bg='white')
    canvas.pack()

    # create an empty PIL image and draw object to draw on
    im = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(im)
    canvas.pack(expand=YES, fill=BOTH)
    canvas.bind("<B1-Motion>", paint)

    # add a button to save the image
    button=Button(text="I'm done!",command=save)
    button.pack()

    master.mainloop()

    im = im.convert('1')
    width, height = im.size
    pix = np.array(np.array(im.getdata())).reshape((height, width))

    n_points_drawing = 0

    for i in range(width):
        if np.min(pix[:,i])==0:
            n_points_drawing += 1


    start_drawing = 0
    stop_drawing = 1
    x_drawing = np.linspace(start_drawing , stop_drawing , n_points_drawing)
    y_drawing = np.zeros( n_points_drawing )
    i_x = 0
    for i in range(width):
        if np.min(pix[:,i])==0:
            x_drawing[i_x] = i
            y_drawing[i_x] = height-np.argmin(pix[:,i])
            i_x+=1
    x_drawing = (x_drawing - min(x_drawing) )/max(x_drawing)
    y_drawing =  10*y_drawing/max(y_drawing)
    
    
    if smoothen:
        x = np.linspace(min(x_drawing) , max(x_drawing) , n_points)
        f_x = interp1d(x_drawing , savgol_filter(y_drawing,51,4) , kind='cubic')
        y = f_x(x)
        x_drawing = x
        y_drawing = y
    
    return np.array([x_drawing, y_drawing])

def make_side_by_side(freqs = 1,amps = 1,phase_offsets = 0,period = 0, duration = 0, n_t = 10000, n_f = 10000, max_freq = 0, max_amp = 0, show_pure = 1,i_iter = 0, rectangles = True, thick_single_line = False, title = False, line_alphas = 1):
    fig, ax = plt.subplots(1,2,figsize=[14,2],gridspec_kw={'width_ratios': [5, 1]})
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    freqs = np.array(freqs)
    amps = np.array(amps)
    phase_offsets = np.array(phase_offsets)
    line_alphas = np.array(line_alphas)
    
    if len(freqs.shape) == 0:
        freqs = np.array([freqs])
        
    if len(amps.shape) == 0:
        amps = np.array([amps])
    
    if len(phase_offsets.shape) == 0:
        phase_offsets = np.array([phase_offsets])
    
    if len(line_alphas.shape) == 0:
        line_alphas = np.array([line_alphas])
    
    if period == 0:
        period = 1 / freqs[0]
        
    if duration == 0:
        duration = 10 * period
    
    if amps.shape[0] != freqs.shape[0]:
        if amps.shape[0] != 0 and amps.shape[0] != 1 and not np.iscomplexobj(amps):
            print('Incorrect shape for amplitudes array, setting random amplitudes.')
        amps = np.ones(freqs.shape[0])
        for i_freq in range(freqs.shape[0])[1:]:
            amps[i_freq] = 0.9 * random.random()
 
    if phase_offsets.shape[0] != freqs.shape[0]:
        if phase_offsets.shape[0] != 0 and phase_offsets.shape[0] != 1:
            print('Incorrect shape for phase offset array, setting random phases.')
        phase_offsets = np.zeros(freqs.shape[0])
        for i_freq in range(freqs.shape[0]):
            phase_offsets[i_freq] = random.random()
     
    if line_alphas.shape[0] != freqs.shape[0]:
        if line_alphas.shape[0] != 0 and line_alphas.shape[0] != 1:
            print('Incorrect shape for line opacity array, setting them all to 1.')
        line_alphas = np.ones(freqs.shape[0])
    
    if max_freq == 0:
        if freqs.shape[0] == 1:
            max_freq = 5 * np.min(freqs)
        else:
            max_freq = np.max(freqs) + np.min(freqs)
    
    
    if freqs.shape[0] == 1:
        show_pure = 0
    
    xt = np.linspace(0 , duration , n_t)
    yt = np.zeros(n_t)
    
    xf = np.linspace(0 , max_freq, n_f)
    yf = np.zeros(n_f) 
    
    peak_width = max_freq / n_f / 2 
    
    for i_freq in range(freqs.shape[0]):
        
        if np.iscomplexobj(amps):
            yt += (1 / len(xt) ) * ( np.real(amps[i_freq]) * np.cos( 2 * np.pi * freqs[i_freq] * xt ) - np.imag(amps[i_freq]) * np.sin( 2 * np.pi * freqs[i_freq] * xt ) )
            yf += (1 / len(xt) ) * np.abs(amps[i_freq]) * np.exp( -(freqs[i_freq] - xf) ** 2 / peak_width )
            ax[1].plot(freqs[i_freq],(1 / len(xt) ) * np.abs(amps[i_freq]),'o',markersize=10,zorder = 1,color = colors[(i_iter + i_freq)%len(colors)],alpha = line_alphas[i_freq])
        else:
            yt += amps[i_freq] * np.sin( 2 * np.pi * freqs[i_freq] * (xt+phase_offsets[i_freq]) )
            yf += np.abs(amps[i_freq]) * np.exp( -(freqs[i_freq] - xf) ** 2 / peak_width )
            ax[1].plot(freqs[i_freq],np.abs(amps[i_freq]),'o',markersize=10,zorder = 1,color = colors[(i_iter + i_freq)%len(colors)],alpha = line_alphas[i_freq])

        if show_pure and freqs.shape[0] > 1:
            if np.iscomplexobj(amps):
                ax[0].plot( xt,(1 / len(xt) ) * ( np.real(amps[i_freq]) * np.cos( 2 * np.pi * freqs[i_freq] * xt ) - np.imag(amps[i_freq]) * np.sin( 2 * np.pi * freqs[i_freq] * xt ) ), color = colors[(i_iter + i_freq)%len(colors)], alpha = line_alphas[i_freq] )
            else:
                ax[0].plot( xt,amps[i_freq] * np.sin( 2 * np.pi * freqs[i_freq] * (xt+phase_offsets[i_freq]) ), color = colors[(i_iter + i_freq)%len(colors)], alpha = line_alphas[i_freq] )
    if show_pure:    
        ax[0].plot(xt,yt,color = 'k', linewidth = 3)
    else:
        if freqs.shape[0]==1 and thick_single_line:
            ax[0].plot(xt,yt,color = 'k', linewidth = 3)
        else:
            ax[0].plot(xt,yt,color = colors[(i_iter + i_freq)%len(colors)] )
        
    ax[1].plot(xf,yf,color = 'k', zorder=0)
                                    
    
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Pressure change')
    ax[0].set_xlim([0,duration])
    if max_amp == 0:
        if np.iscomplexobj(amps):
            max_amp = (1 / len(xt) ) * np.max(np.abs(amps))
        else:
            max_amp = np.max(np.abs(yt))
        ax[1].set_ylim([-0.05 * max_amp,max_amp*1.1])
        ylim = 1.1 * np.max(np.abs(yt))
    else:
        ax[1].set_ylim([-0.05 * max_amp,1.1 * max_amp])
        ylim = 1.1 * max_amp
    ax[0].set_ylim([-ylim,ylim])
    rect_opacity= 0.1
    if rectangles:
        for i_period in range(int(duration / period)):
            if i_period%2==0:
                rect = Rectangle([i_period*period,-ylim],period,2*ylim,alpha=rect_opacity,color='k',linewidth = 0)
                ax[0].add_patch(rect)
    if title:
        ax[0].set_title(title)
    else:
        if freqs.shape[0] == 1:
            ax[0].set_title('Pure tone, frequency = '+ "{:.2f}".format(freqs[0]) +' Hz')
        else:
            ax[0].set_title('Some complex tone')

        
        
    ax[1].set_xlim([0,max_freq])
    
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title( '\"Frequency representation\"')
    
    
def make_complex_tone(n_tones = 3, lowest_freq = 1., pitched = True):
    
    if float(lowest_freq)==0:
        print("Invalid lowest frequency. Setting it to default (1).")
        lowest_freq = 1.
    elif float(lowest_freq)<0:
        print("Lowest freq is negative. Taking the absolute value.")
        lowest_freq = -1 * float(lowest_freq)
    else:
        lowest_freq = float(lowest_freq)
    
    if int(n_tones)<=0:
        print("Invalid number of pure tones. Setting to default (3).")
        n_tones = 3
    else:
        n_tones = int(n_tones)
    
    freqs = lowest_freq * np.ones(n_tones)
    amps = np.ones(n_tones)
    phase_offsets = np.ones(n_tones)
    
    mock_n_points = 1000
    mock_xt = np.linspace(0,10 * 1/freqs[0], mock_n_points)
    mock_yt = np.zeros(mock_n_points)
    
    for i_tone in range(n_tones):
        if i_tone>0:
            freqs[i_tone] = pitched * (i_tone + 1) * lowest_freq + (1 - pitched) * (i_tone + 0.3 + random.random()*0.7) * lowest_freq
            amps[i_tone] = 0.3 + 0.6 * random.random()
        phase_offsets[i_tone] = random.random()
        mock_yt += amps[i_tone] * np.sin( 2 * np.pi * freqs[i_tone] * (mock_xt+phase_offsets[i_tone]) )
    
    max_amp = np.max(np.abs(mock_yt))
    
    for i_tone in range(n_tones):
        make_side_by_side(freqs[i_tone],amps[i_tone],phase_offsets[i_tone],duration = 10 * 1 / freqs[0],max_amp = max_amp,max_freq = freqs[-1] + freqs[0], i_iter = i_tone)
    make_side_by_side(freqs,amps,phase_offsets,duration = 10 * 1 / freqs[0],max_amp = max_amp,max_freq = freqs[-1] + freqs[0],rectangles = pitched)
    
    
def process_drawing(drawing = np.zeros([1,1]), n_freqs_to_plot = 10, one_by_one=True, start = 0, stop = 1, n_points = 10000):
    
    print(drawing.shape)
    if drawing.shape == (1,1):
        print("You didn't draw anything! Doing a straight line.")
        drawing = np.array([np.linspace(0,1,10000),np.linspace(-1,1,10000)])

    if int(n_freqs_to_plot)<=0:
        print("Invalid number of pure tones. Setting to default (10).")
        n_freqs_to_plot = 10
    else:
        n_freqs_to_plot = int(n_freqs_to_plot)


    x = drawing[0,:]
    y = drawing[1,:]    

    xf = np.fft.fftfreq(n_points,(stop - start) / n_points)
    yf = np.fft.fft( y )

    fig, ax = plt.subplots(1,2,figsize=[14,2],gridspec_kw={'width_ratios': [5, 1]})

    ax[0].plot(x,y - np.real(yf[0])* (1 / len(x) ),linewidth =3,color = 'k')
    ax[0].set_title('Original complex tone')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Pressure change')
    ax[0].set_xlim([0,1])



    ax[1].set_xlim([0,1])
    ax[1].set_ylim([0,1])
    ax[1].text(0.5,0.5,'?',ha='center',va='center',fontsize=50)
    ax[1].set_xlabel( 'Frequency (Hz)' )
    ax[1].set_ylabel( 'Amplitude' )
    ax[1].set_title( '\"Frequency representation\"')


    
    make_side_by_side(freqs = xf[1:n_freqs_to_plot+1],amps = yf[1:n_freqs_to_plot+1],period = 0, duration = stop-start, n_t = 5000, n_f = 5000, max_freq = 0, max_amp = 0, show_pure = 1,i_iter = 0, rectangles = False, title = 'Approximation of original complex tone as a sum of ' + str(n_freqs_to_plot) + ' pure tones')

    if one_by_one:
        for j in range(n_freqs_to_plot+1)[1:]:
            line_alphas = 0.3 * np.ones(j)
            line_alphas[-1] = 1
            if j == 1:
                make_side_by_side(freqs = xf[1:j+1],amps = yf[1:j+1],line_alphas = line_alphas, period = 0, duration = stop-start, n_t = 5000, n_f = 5000, max_freq = 0, max_amp = 0, show_pure = 1,i_iter = 0, rectangles = False, thick_single_line = True, title = 'Approximation of original complex tone as a pure tone')
            else:
                make_side_by_side(freqs = xf[1:j+1],amps = yf[1:j+1],line_alphas = line_alphas, period = 0, duration = stop-start, n_t = 5000, n_f = 5000, max_freq = 0, max_amp = 0, show_pure = 1,i_iter = 0, rectangles = False, thick_single_line = True, title = 'Approximation of original complex tone as a sum of ' + str(j) + ' pure tones')
