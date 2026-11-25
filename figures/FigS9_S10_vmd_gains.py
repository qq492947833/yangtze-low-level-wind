from figure_paths import DATA, RESULTS, WORK, FIGURES, setting
from pathlib import Path
import csv
import hashlib
import re
import numpy as np
COMP=['u','v','omega'];LEVELS=[850,925,1000];STAGES=['stage1','final_residual'];METRICS=['delta_R','delta_RMSPE']
SOURCE=RESULTS/'VMD_common_time_spatial_gains.csv'
DOCX_WIDTH_MM=146.40277777777777
FIGSIZE=(7.2,3.8);NOMINAL_WIDTH_MM=FIGSIZE[0]*25.4
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def read_new_source():
 digest=sha(SOURCE)
 with SOURCE.open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);required={'stage','lead_hour','component','level_hpa',*METRICS}
  assert required.issubset(reader.fieldnames or []),reader.fieldnames
  rows=list(reader)
 assert len(rows)==108,('expected exactly 108 rows',len(rows))
 data={};source_map=[]
 for i,r in enumerate(rows,2):
  lead=float(r['lead_hour']);level=float(r['level_hpa']);assert lead.is_integer() and level.is_integer()
  key=(r['stage'],int(lead),r['component'],int(level))
  assert key not in data,('duplicate key',key)
  vals={m:float(r[m]) for m in METRICS};assert all(np.isfinite(v) for v in vals.values()),key
  if 'evaluation_scope' in r:assert 'common' in r['evaluation_scope'].lower(),('unexpected scope',r['evaluation_scope'])
  data[key]=vals;source_map.append({'csv_line':i,'stage':key[0],'lead_hour':key[1],'component':key[2],'level_hpa':key[3],**vals})
 expected={(s,h,c,l) for s in STAGES for h in range(1,7) for c in COMP for l in LEVELS}
 assert set(data)==expected,{'missing':list(expected-set(data)),'unexpected':list(set(data)-expected)}
 assert sha(SOURCE)==digest,'CSV changed during read'
 limits={m:max(abs(v[m]) for v in data.values()) for m in METRICS}
 scales={m:(v if v>0 else 1.) for m,v in limits.items()}
 return data, source_map, digest, limits, scales


def render():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
 from PIL import Image
 data,source_map,digest,raw_limits,limits=read_new_source()
 out=FIGURES
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5,'savefig.facecolor':'white'})
 results=[]
 for stage,stem in [('final_residual','FigS9'),('stage1','FigS10')]:
  fig=plt.figure(figsize=FIGSIZE)
  gs=fig.add_gridspec(2,4,width_ratios=[1,1,1,.06],left=.085,right=.935,bottom=.125,top=.9,wspace=.33,hspace=.66)
  caxes=[];annotations=[];mapping=[];colorbars=[]
  for i,metric in enumerate(METRICS):
   cmap=LinearSegmentedColormap.from_list('common_time_gain',['#c45a3a','#ffffff','#2b6fb0'],N=257)
   limit=limits[metric];assert np.isfinite(limit) and limit>0
   exponent=int(np.floor(np.log10(limit)));scale=10.**exponent
   exp_text=str(exponent).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
   for j,c in enumerate(COMP):
    ax=fig.add_subplot(gs[i,j]);panel=chr(97+3*i+j)
    values=np.array([[data[stage,h,c,lev][metric] for h in range(1,7)] for lev in LEVELS])
    assert np.isfinite(values).all() and np.abs(values).max()<=limit
    im=ax.imshow(values,cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto',interpolation='nearest')
    ax.set_xticks(range(6),labels=[str(h) for h in range(1,7)]);ax.set_yticks(range(3),labels=[str(l) for l in LEVELS]);ax.set_xlabel('Lead time (h)')
    if j==0:ax.set_ylabel('Pressure (hPa)')
    if i==0:ax.set_title('ω' if c=='omega' else c,fontsize=10,pad=13)
    ax.annotate(f'({panel})',xy=(0,1),xycoords='axes fraction',xytext=(0,5),textcoords='offset points',ha='left',va='bottom',weight='bold',fontsize=9)
    for y,lev in enumerate(LEVELS):
     for x,h in enumerate(range(1,7)):
      v=float(values[y,x]);label_text=format(v/scale,'.2g') if v else '0'
      if v!=0:assert float(label_text)!=0,'Nonzero value rounded to zero'
      text_color='white' if abs(v)/limit>.64 else '#111111'
      text=ax.text(x,y,label_text,ha='center',va='center',fontsize=6.6,color=text_color)
      record={'figure':stem,'stage':stage,'panel':panel,'metric':metric,'component':c,'level_hpa':lev,'lead_hour':h,'raw_value':v,'display_scale':scale,'display_exponent':exponent,'display_value':v/scale,'display_text':label_text,'rendered_rounded_raw_value':float(label_text)*scale,'color_limit':limit,'normalized_color_coordinate':float(im.norm(v)),'text_color':text_color}
      annotations.append((text,record));mapping.append(record)
    ax.tick_params(length=2)
   cbax=fig.add_subplot(gs[i,3]);caxes.append(cbax);ticks=np.linspace(-limit,limit,5)
   cb=fig.colorbar(im,cax=cbax,ticks=ticks)
   cb.set_label(('ΔR' if i==0 else 'ΔRMSPE')+f' (×10{exp_text})',fontsize=8,labelpad=7,fontfamily='DejaVu Sans')
   ticktexts=[format(v/scale,'.2g') for v in ticks];cb.ax.tick_params(labelsize=7,length=2);cb.set_ticklabels(ticktexts)
   colorbars.append({'metric':metric,'source_row_count':108,'actual_max_absolute_delta':raw_limits[metric],'symmetric_limits':[-limit,limit],'display_scale':scale,'display_exponent':exponent,'tick_values':ticks.tolist(),'tick_texts':ticktexts,'zero_only_fallback':raw_limits[metric]==0})
  fig.canvas.draw();renderer=fig.canvas.get_renderer()
  for text,record in annotations:
   available=text.axes.get_window_extent(renderer).width/6-2.
   for attempt in range(4):
    width=text.get_window_extent(renderer).width
    if width<=available:break
    fitted=text.get_fontsize()*available/width*.97
    if fitted<5.:raise ValueError('Annotation cannot fit at >=5pt; revise this new plot only')
    text.set_fontsize(fitted);fig.canvas.draw()
   assert text.get_window_extent(renderer).width<=available
  fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.svg'));fig.savefig(out/(stem+'.png'),dpi=800)
  plt.close(fig)
  with Image.open(out/(stem+'.png')) as src:
   rgb=src.convert('RGB');pixels=rgb.size;pixel_hash=hashlib.sha256(rgb.tobytes()).hexdigest();rgb.save(out/(stem+'.tiff'),compression='tiff_lzw',dpi=(800,800));rgb.close()

if __name__ == "__main__":
    render()
