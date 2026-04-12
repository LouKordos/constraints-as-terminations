set nocompatible

" Bootstrap vim-plug 
if empty(glob('~/.vim/autoload/plug.vim'))
    silent !curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
endif
runtime autoload/plug.vim
" Plugin Definitions
call plug#begin('~/.vim/plugged')
    Plug 'tpope/vim-surround'
    " Plug 'preservim/nerdcommenter'
    Plug 'tpope/vim-commentary'
    Plug 'itchyny/lightline.vim'
    Plug 'sainnhe/gruvbox-material'
    Plug 'sheerun/vim-polyglot'
    Plug 'jasonccox/vim-wayland-clipboard'
    Plug 'ojroques/vim-oscyank', {'branch': 'main'}
call plug#end()
" Auto install missing plugins on startup
autocmd VimEnter * if len(filter(values(g:plugs), '!isdirectory(v:val.dir)')) | PlugInstall --sync | source $MYVIMRC | endif

" Fix lightline display
set laststatus=2

" For commenting using vim-commentary
filetype plugin on
filetype plugin indent on

if has('termguicolors')
    set termguicolors
endif
" RGB fallback just for screen*/tmux* terms
if !has('gui_running') && &term =~# '^\%(screen\|tmux\)'
    let &t_8f = "\<Esc>[38;2;%lu;%lu;%lum"
    let &t_8b = "\<Esc>[48;2;%lu;%lu;%lum"
endif

set background=dark
let g:gruvbox_material_background = 'soft'
let g:gruvbox_material_better_performance = 1
" === Gruvbox autoload ===
function! s:TryGruvbox() abort
    if !empty(globpath(&rtp, 'colors/gruvbox-material.vim'))
        " File exists – safe to load the theme exactly once
        colorscheme gruvbox-material
        augroup GruvboxOnce | autocmd! | augroup END
    endif
endfunction
" Run immediately and after any PlugInstall
call s:TryGruvbox()
augroup GruvboxOnce
    autocmd! User PlugLoaded ++once call s:TryGruvbox()
augroup END

" https://stackoverflow.com/questions/33380451/is-there-a-difference-between-syntax-on-and-syntax-enable-in-vimscript
if !exists("g:syntax_on")
    syntax enable
endif

set tabstop=4
set shiftwidth=4
set expandtab
set mouse=
set timeoutlen=1000
set ttimeoutlen=50
set hlsearch
set incsearch
set smartcase
" This is required to temporarily turn off smartindent when wanting to paste something using insert mode + ctrl-shift-v.
" The reason this might be needed is system clipboard not being synced with vim registers for various reasons, so while
" using P or p is better, sometimes its required. By pressing F2 before insert mode => ctrl-shift-v, smart features are disabled
" and pasting can proceed normally. then press f2 again to disable paste mode.
set pastetoggle=<F2>

" ============================================================================
" Clipboard routing policy: "Always On" Strategy
" Rationale:
"   We deliberately choose the simpler approach of always enabling the system 
"   clipboard ('unnamedplus'). This avoids complex environment detection scripts 
"   while satisfying the primary requirement: seamless pasting in Local TMUX.
"
" Scenarios & Behavior:
"   1) Local Desktop (inc. TMUX):
"      - Paste ("+p): Works flawlessly (reads local OS clipboard).
"      - Yank (y): Writes to local OS clipboard directly.
"        * Note: OSC52 also fires (redundant "double copy"), which is benign.
"
"   2) Remote SSH (Headless/No X11):
"      - Paste: Clipboard connection fails gracefully; falls back to 'viminfo'.
"      - Yank: 'unnamedplus' is ignored. OSC52 handles the transfer to client.
"      - Persistence: Works via 'viminfo' (yank -> quit -> reopen -> paste).
"
"   3) Remote SSH (With X11/GUI clipboard):
"      - Trade-off: We connect to the remote machine's physical clipboard.
"        * Privacy: Yanks overwrite the remote clipboard (acceptable for single-user servers).
"        * Paste: May paste remote content instead of 'viminfo' history.
"        * Safety Net: If the default paste is wrong, use "0p to recover the 
"          last yank from the viminfo-persisted "0 register.
"
" Conclusion:
"   We accept the minor privacy/startup risks on GUI-enabled remote servers 
"   in exchange for guaranteed clipboard functionality in local workflows 
"   without fragile detection logic.
" ============================================================================

" Always enable system clipboard integration. This ensures "+p works inside local TMUX sessions.
set clipboard=unnamedplus,unnamed

" We still need OSC52 for two cases:
" a) True Remote SSH (where system clipboard is on the server, not the client).
" b) Local TMUX attached from afar (system clip hits desktop; OSC52 hits laptop).
let g:oscyank_max_length = 0
let s:IsRemoteSsh = exists('$SSH_TTY') || exists('$SSH_CONNECTION')

function! s:VimOsc52MaybeCopy(ev) abort
    " Trigger OSC52 if we are in any tmux session or any ssh session.
    if !(exists('$TMUX') || s:IsRemoteSsh)
        return
    endif
    
    if !exists('*OSCYankRegister')
        return
    endif
    if index(['y', 'd', 'c'], get(a:ev, 'operator', '')) == -1
        return
    endif
    let l:reg = get(a:ev, 'regname', '')
    if empty(l:reg)
        let l:reg = '"'
    endif
    if index(['"', '+', '*'], l:reg) == -1
        return
    endif
    call OSCYankRegister(l:reg)
endfunction

augroup VimOsc52OnYank
    autocmd!
    autocmd TextYankPost * call s:VimOsc52MaybeCopy(v:event)
augroup END

" When no wl-copy or clipboard is present (e.g. on a server), vim uses .viminfo for pasting across sessions. Increase size to avoid truncating
" '200    : Remember marks/cursor position for the last 200 edited files
" f1      : Store global file marks (A-Z)
" <       : Allow registers with up to this many lines to be saved
" s       : max size per register in KiB (default is s10, which is too small for big yanks)
" :100    : Remember the last 100 command-line commands
" /100    : Remember the last 100 search history items
" h       : Disable search highlighting when Vim starts
set viminfo='200,f1,<30000,s5000,:100,/100,h

" smartundo (more granular, see https://vi.stackexchange.com/questions/2376/how-to-change-undo-granularity-in-vim)
function! s:start_delete(key)
    let l:result = a:key
    if !s:deleting
        let l:result = "\<C-G>u".l:result
    endif
    let s:deleting = 1
    return l:result
endfunction

function! s:check_undo_break(char)
    if s:deleting
        let s:deleting = 0
        call feedkeys("\<BS>\<C-G>u".a:char, 'n')
    endif
endfunction

augroup smartundo
    autocmd!
    autocmd InsertEnter * let s:deleting = 0
    autocmd InsertCharPre * call s:check_undo_break(v:char)
augroup END

inoremap <expr> <BS> <SID>start_delete("\<BS>")
inoremap <expr> <C-W> <SID>start_delete("\<C-W>")
inoremap <expr> <C-U> <SID>start_delete("\<C-U>")

" Use triple quotes for python block commenting:
augroup python_block_comments
  autocmd!
  " 1) default single-line comments with #
  autocmd FileType python setlocal commentstring=#\ %s
  " 2) in Visual mode, gc calls the block-comment function
  autocmd FileType python vnoremap <buffer> gc :call PythonBlockComment()<CR>gv
augroup END

function! PythonBlockComment() range
  let l:start = a:firstline
  let l:end   = a:lastline
  let l:ind   = indent(l:start)
  " insert opening ''' above selection
  call append(l:start - 1, repeat(' ', l:ind) . "'''")
  " insert closing ''' below selection
  call append(l:end   + 1, repeat(' ', l:ind) . "'''")
endfunction

" Maps Ctrl+a to go to the first non-whitespace character in Insert mode
inoremap <C-a> <C-o>^

" Support home and end buttons for navigating lines
inoremap <End> <C-o>$
inoremap <Home> <C-o>0
vnoremap <End> $
vnoremap <Home> 0

" Center after moving to next or previous search result
nnoremap n nzzzv
nnoremap N Nzzzv

nnoremap <C-Del> dw
inoremap <C-Del> <C-o>dw
inoremap <C-BS> <C-W>

" Center cursor on screen after scrolling
nnoremap <C-d> <C-d>zz
nnoremap <C-u> <C-u>zz

" Keep visual mode after indenting
:vnoremap < <gv
:vnoremap > >gv

" Detect the [Z sequence as <S-Tab>
exe 'set t_kB=' . nr2char(27) . '[Z'
" Allow indenting with tab and shift-tab
xnoremap <Tab> >gv
xnoremap <S-Tab> <gv

" Move lines like VS Code
" Supports: Alt+Up / Alt+Down and Alt+K / Alt+J
nnoremap <silent> <Plug>(MoveLineUp)   :m .-2<CR>==
nnoremap <silent> <Plug>(MoveLineDown) :m .+1<CR>==
inoremap <silent> <Plug>(MoveLineUp)   <Esc>:m .-2<CR>==gi
inoremap <silent> <Plug>(MoveLineDown) <Esc>:m .+1<CR>==gi
vnoremap <silent> <Plug>(MoveLineUp)   :m '<-2<CR>gv=gv
vnoremap <silent> <Plug>(MoveLineDown) :m '>+1<CR>gv=gv

" Alt+Arrow (terminal sends 1;3A/B)
execute "set <F31>=\e[1;3A"
execute "set <F32>=\e[1;3B"

nmap <F31> <Plug>(MoveLineUp)
nmap <F32> <Plug>(MoveLineDown)
imap <F31> <Plug>(MoveLineUp)
imap <F32> <Plug>(MoveLineDown)
vmap <F31> <Plug>(MoveLineUp)
vmap <F32> <Plug>(MoveLineDown)

"  Alt+J/K as Meta (terminal sends ^[j / ^[k)
execute "set <M-j>=\ej"
execute "set <M-k>=\ek"

nmap <M-k> <Plug>(MoveLineUp)
nmap <M-j> <Plug>(MoveLineDown)
imap <M-k> <Plug>(MoveLineUp)
imap <M-j> <Plug>(MoveLineDown)
vmap <M-k> <Plug>(MoveLineUp)
vmap <M-j> <Plug>(MoveLineDown)
