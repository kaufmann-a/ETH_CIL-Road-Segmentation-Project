## Tmux guide

- Enter tmux: `tmux`
- Split panes left/right: `Ctrl+b %`
- Split panes top/bottom: `Ctrl+b "`
- Switch panes: `Ctrl+b ArrowKeys`
- Closing panes: `Ctrl+d` or `exit` (by clossing all panes you exit the session)
- Detach Session (session keeps running in background): `Ctrl+b d`
- Reattach to running session: Check out running sessions with `tmux ls` then attach to session with `tmux attach -t 0`
  where 0 is session 0.
- Enter scroll-mode (to scroll up and down): `Ctrl+b [` then arrow keys (to quit press `q`)

### Other commands

- New window: `Ctrl+b c`
- Switch windows back and forth: `Ctrl+b p`, `Ctrl+b n` or `Ctrl+b WindowNr`
- Toggle a pane fullscreen and embeded: `Ctrl+b z`

## Preprocessing

### Prepare additional Training data on cluster scratch

1. Connect to leonhard and navigate to the scratch `cd /cluster/scartch/...`
2. load a newer version of curl: `module load curl`
3. Open your browser (chrome, firefox) and navigate to oneDrive
4. Open developer tools: `CTRL+SHIFT+I`
5. Navigate to the  "Networks" tab and enter "zip?" in the filter (networkstack might now be empty)
6. Select all folders you like to download and click on "Download"
7. cancel the download
8. Now there should be one entry in the Networks tab
9. Right click on it: Copy -> Copy as cURL
10. Paste the command into your leonhard terminal and add `-o data.zip` to specify the output file
11. Hit enter and wait
12. Unzip the data `unzip data.zip`  (with -d to add target directory path)

### Transformations

1. cd to location of augmentations script `cd /cluster/home/{username}/cil-road-segmentation/preprocessing`
2. Load environment
3. run augmentation.py, add location of data as argument: `python augmentations.py "/cluster/scratch/{username}/data"`
