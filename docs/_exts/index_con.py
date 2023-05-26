from m2r2 import convert
import os

class IndexCon:
    
  def __init__(self, source, output):
    self.source = source
    self.output = output
    self.preprocess()

  def preprocess(self):
    with open(self.source, 'r') as f:
      
      # remove the CI link from the file
      lines = f.readlines()
      lines = [line for line in lines if '[CI]' not in line]

      # change local links to the ones related to the _build/html directory and extension to .html
      lines = [line.replace('](docs/', '](') for line in lines] 
      lines = [line.replace('.md)', '.html)') for line in lines]

      # get the raw text within the <details> tag
      start_details_tag = [line for line in lines if '<details>' in line]
      summary_tag = [line for line in lines if '<summary>' in line]
      end_details_tag = [line for line in lines if '</details>' in line]
      start_details = lines.index(start_details_tag[0])
      summary_line = lines.index(summary_tag[0])
      end_details = lines.index(end_details_tag[0])

      before = convert(''.join(lines[:start_details-1]))
      end = convert(''.join(lines[end_details+1:]))

      collapse_rst = lines[summary_line+1:end_details]
      collapse_rst = [ "**" + x.split("# ")[1][:-1] + "**\n" if '# ' in x else x for x in collapse_rst]

      # print(collapse_rst)

      collapse_rst = convert(''.join(collapse_rst))
      collapse_rst = collapse_rst.split("\n")
      collapse_rst = [ '    ' + x for x in collapse_rst]

      collapse_rst = [f'\n.. collapse:: Click to SHOW examples\n'] + collapse_rst
      os.remove(self.output)

      with open(self.output, 'a') as f:
        f.write(before)
        f.write('\n'.join(collapse_rst))
        f.write(end)
        f.write('\n\n')
        
        with open('toc.bak', 'r') as t:
          f.write(t.read())

if __name__ == '__main__':
  index = IndexCon('../../README.md')

