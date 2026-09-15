#!/usr/bin/env python3
"""User-run continuation for the existing PUBLIC commodity publication branch.

--prepare: resume the documented partial lint correction, preserve all 120
applied safe fixes, correct UP022 in REVIEW COPIES, apply the full configured
safe lint fixes to scratch candidates, then require unchanged quality gates.
--publish --reviewed-public-code: stage the reviewed manifest, commit, push,
and open/reuse a pull request when the GitHub CLI is available and authenticated.
--merge: request a normal, head-pinned PR merge only after successful checks.
--verify-merge: fetch main and verify the published file hashes.

No research training, package installation, force push, branch reset, source
checkout replacement, repository visibility change, or cloud operation.
Prepared by static source review; tests and quality run ONLY when the user runs it.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import copy
import datetime
import difflib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import zipfile
import zlib
from pathlib import Path, PurePosixPath

ROOT = Path.home() / "projects/commodity-prediction-manual"
REVIEW = Path.home() / "projects/commodity-prediction-review"
PYTHON = Path.home() / "projects/commodity-prediction-current/.venv/bin/python"
STORE = Path.home() / "commodity-public-update/continuation-20260914"
REPORT = Path.home() / "commodity_publish_report.json"
BUNDLE = Path.home() / "commodity_publish_report.zip"
REPO = "alvaromendizabal/commodity-prediction"
BRANCH = "chore/public-research-update-20260913"
PIN = "d142a4cb57a5c4b2880f9341619e13a735b1cddc"
PLOT = "application/vnd.plotly.v1+json"
MAX_FILE = 30 * 1024**2
MAX_LOG = 32 * 1024**2
MAX_STATE = 8 * 1024**2
# Historical selector: ONLY for reproducing the already-applied 120 fixes.
FORMAT_SELECT = "I,F401,UP017,UP022"
MAX_CANDIDATE_PASSES = 2
STATE_FILE = STORE / "state.json"
TITLE = "Publish reproducible commodity feature research and executed portfolio"
ARCHIVE_ROOT = "docs/history/manual_execution_sources"
PUBLIC_PROVENANCE = "reports/manual_research/publication_source_map.json"
PUBLIC_GUIDE = "docs/PUBLICATION_REPRODUCIBILITY.md"
COMPARE_URL = "https://github.com/" + REPO + "/compare/main..." + BRANCH + "?expand=1"
SECRETS = [
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r'(?i)(?:X-Amz-Signature|X-Amz-Credential)=[^&\s"<>]{12,}'),
    re.compile(r"(?i)https://[^/\s:@]+:[^/\s@]+@"),
]
PULL_REQUEST_BODY = (
    b"## Public commodity research publication\n"
    b"\n"
    b"- Publish the actual manual research source, tests, declarations, aggregate evidence and saved Plotly notebooks.\n"
    b"- Reuse the existing review branch and preserve the research execution checkout.\n"
    b"- Preserve original execution bytes, models, data and all three edited notebooks.\n"
    b"- Resolve the known subprocess capture warning without unsafe bulk fixes or rule suppression.\n"
    b"- Local repository quality and original publication verifiers passed for the recorded bytes; require CI on this new commit before merging.\n"
    b"\n"
    b"No private forecasting fits or final-test evaluation were performed for publication.\n"
)

# Exact evidence and targeted source edits, encoded as inert JSON below.
SPEC = None
RUN_REPORT = {}
LOGS = []
DEADLINE = float("inf")

RECOVERY_VERSION = "publication-configured-lint-20260914"
PRIOR_PUBLISHER_SHA = "4df170ddcc2bc28b50ef7222d12b12581d61229729fe23f8e018c5497a2cffff"
PRIOR_RUN_ID = "20260914T173427180877Z"
PRIOR_STATE_SHA = "5ed67bfd76c6d2328fe9bc8bb59a8765499a79daf0ab4d08e60b2d5c9c45aea0"
PRIOR_REPORT_SHA = "89dac9c2e3681d797440a47b0d3462adfbc6d866de6c144d4a745fb6832f704a"
PRIOR_LINT_SHA = "6dbcc21240526b50af72b05de6472dea56c0e09ad41d2aa31cd34cda7400d2f6"
# Compressed inert source is used for exact reconstruction, never executed.
ORIGINAL_PUBLISHER_B85 = (
    "c-rl~X?G*Xkp=plzoJGyegI?;tbL&do}t)~non$!!^K;7Pq~&X5^fNHQ9yT7ZT|OnBbQo$Dl|w@_V}Eq@kj(vl}ltSHzFe=f"
    "Ab#)t9)?~kEaK0`ca!dEbnGhYww$HzPZWSVt=ukYRPQ698Xu#ay*-A*=(UL@0i9uk8^x`t6kiDe|mC^hwkrZ>3I2|%~$bcoQ"
    "S96MKn$B277z^`||~xM+<hSEf^Y1#uLWd$yf9FWXw`6F!wY1GG9hZ*4OYj`v1tZ*<yS<o<@_rk3JT7dAavtI{P%$CgbT+OIe"
    "0*a&0`-E=PYj8U0Z^KEF5_UG=qSnrbxq+F<{_iY7G1TlA3+XjC-V{7&27U$Bp3_KBt9t7Jc!r5G$mb1Md_e%J1!={RHgvd`x"
    "{#((Fl{I0)89nR)#daz)toM{oho=ouC-&c58`*g>|SbiKYU$0{A`1C{@=UVhJ8c(7a^QPCM6`laGK|Pkz@b1~-mXl005^2-f"
    ";y#-6wL2DJq~qxn1G>1>_;cE9I(g8dY{?cNYm%@$&sGyHxns!(KG~0KG0q-TqZwN!cbE-GM}3PQ`4A;h&@^9>y9j@yfxWYtM"
    "ynDn@ZNHP7eP|YK1x0ik;YRHYck;r-zPpvn4pa00{$+jmCsiAX^#(Uwp!xr`6No%Jtp3#53_u{oGl)-kK=qCkNI*XchU3~Oa"
    "b0YCbJc%^MZFY*t-yXm1^+==bdpPdyp&kora|bhtM3=@)KX2pPl}>TotU?g1!bPEeFK?d)Z=kuZ3Z@TCNr>47Kt7e73+6OsB"
    "IY?>gVxQ{OFaF@&5IpYx^qlVpc|UQWib`X<HNF30!9!&x$2PSgh?b>-Q=<+G{!dzOnK&7&nfqQ-N9kA3X|W4r+2KGR3_#-iv"
    "WzgsQGMJM_A7V=blTE+9ljF_eP=Arm=$?oUG1I6dcz`+3ti?R6Xi`8_zBwDM_Kaa(`dza_u*V>Q|b#OPkXWgE5pmose-$-^3"
    "3e`b6A{{40+<izH7OS%-rTU3)i>2@EUHtj__4(OT-%eK0Gt=e4;3HHbSB*j~cJ{8W&o4)t%~e_v%40RBWj?5C&V5riT|F>tn"
    "$yMk<@J_kLMeeEiNj(!AU^2qeSdTI>U8vk*RTMhGrZG@CLbg451XdrpQAXMG)^6zTppbrzor+GyV-&r$O$RAmorn7u|VPzqH"
    "iZPY%{Wx*zqDKv18Ns^(?S#!!-hCSdnKrv5}-{f)`FXH*_LW{K8Hge4M6(`DC`7JPbY>zZ27abM&Y1hZ8<wOV_^D4BfQ9{nq"
    "5)pPv6nUz=-RUtJ$v)AN4i`yb9P-yB_sSEJL>G4J=J|AVa?{hJHj@aW&BN$ZLiRc~UEL0E2zB=N6LuKACh3u!Sx<HampC1X<"
    "FrS;2LL<*x+l%#jqXO^s>(luI^Y&IFsI(tW#$FEQRFbbt`bkbRpAKbxCLH8X9?hnO#AW$e&S`O-Qa~ob<p8sKVhNa?T66>9-"
    "*HqsJ6;mu*2=Al0oRMtl$D5N^qhbK!%cJX)^Rp0~b9w&i=J@3Mlhc#yKM(HH&ff9)n~S5%QFwECO7Gk)m-GDa;NW(=yj#Tsk"
    "l~=C{Z8XN_dBgaJVwg-fUL}5FreQggW=czVxQ-r&d>n;N5_|=>nk+*c2C1U3r20tp*^~b&bzq#Ux!COogDq+=<4LC_x5Q2FZ"
    "zD){)^#0zKc7(e$yMbcW>u20GszAJs$1<%-^u|{+3tx975>V$~)ovFLwJ&`}EKEqaPsx7~<vCk@k<#vG$Mii_zKD)oZQyUl*"
    "4re;}6pY4m6QY-20k{~Gu3`k(fX?tk9DBIQgR^bhg*7_v!_Ae!`sZ~yoId6$3lpTEb{&HiI+TWV3>{_nxN{P4fu|L&0gg+Kn"
    "A-g&?G8*TJCN|qXwB-%~{_E^G-G2&5aJ)shoz+CtpKnT!;FcyPW|9`KbP-8ZkeH!dtU0}LH?QAw>dzUwyV}xS+^0)^P<?H!="
    "@73t&)#=F@F<Ca5MaynyJk2`2y}i9;66LvewVchnm#Zlmm(gM|Tl5b3#9)Ol6bd98E!DC>hyvd+P)Zo^bhY3xN~%7x;Cp*Oh"
    "d@l9y0Ss}xU#^5;tzVJ_(1>q88aHJmPv0g&Sx26u4T6;8_e&bZahmL>WxavJ;1M~<BA6cckFXIzJ*CFUKZr+evBq7R&St7Qa"
    "MOh_w!u*+y}vF>4zp^ro}S+z#j79^@4r@&YFD+r_pry1CTN6kvP4Zc6tNZelgZ416Ryu99a=G`>1*)RDuP`my2%k3;$tG2Br"
    "!py7Z{{ZJcRcz#s84z!?2KA3(x^ukdjWYK2jp&n8fIuub@?LqZRPDWBr4sDJ3^cklSms=;Xb(48&PZqn2KlhiwT8;N?LCx_0"
    "77L9?iIp=h4rZmlBf*c_YI%32r!DeAGIUU5eztLXJ_~5}_v&l!nOgSV7Pz)-3M$`q^hKKP%<M8;QbRhs`87b)jZz<!07T?gL"
    "Z&U6Wd`thKDWk>weK~9puy`M|${+5jk8WjR>yx_@k3a&`F*!ekjNv!X;3JG|4kPrDeq=N{0g2_*Iqjerz{(Jsat+G9IB^b!<"
    "x|yh@)<mJj8)%{mf=t%m0vQDl0ZU<f<qT@WV&+>VkcvGQ$uC;Wq<UOT0r`{U_`#k)$tV!DSsgBDZb_3kk;t-@SiXr|ID=Ci|"
    "(2^>diRMxo-p7F6i-8G)}<sb~T9>+%Xc9BMIUI6C;Zm=*}?WN~hunXaw~&O*)Jp0G8#C!fD%li3Coxs0QBG%Fp+ts={er{cb"
    "#+^|b#OYTepu@%GCmijQRr#aGloGgW>dXHd9eDLkh!*S1j#LN!uCt~JeDu64AW-H#Kw+7%VryvAI2ae@dr>I>>jqkE;P%HIa"
    "}A0QrGDR5Fra*+*ZANWTp6KS9OG_ystpw&?R4c^b`?=GNlHvT;9aHvxB*C*i{R{9gIEf)_Zr9J^Zz)2fqDFp$#)KljZU;|lB"
    "P=-bKbpaCJ4L&W#(3PARO+RN7c+lPYPt-~NFipDZ8B$!cZm)a_Pe`x2Tg+CFPm+NGDwBtrUcx(kf}Co3`QDn`<FVD03tahxT"
    ";9JGvRt=p+_1s$r+K#`?FC-%^$Jyx0+k4Jb6BV_JPppAk^++`u^f;zV)BszT-k2t?d9mz(ed@@RfvaQjn1x5j!xfq`kP9KzE"
    "EZ|iKcuF`Wk1inu(<E2PxxrwX<4g`+lboPO?`;?;LZB;kS&uMX+FMbKyo5V2|tT#cV#y3$0qwIYI^`qJWdS><TXZ_S<aCCTX"
    "tZYB9!76<qNJ<4-z=0uzBh`MdPF_Vy!w;(S*ToD$u=?Fbhz4XLg7AP{eD<0(M9Whgq}51|X8>Vy&yk12fFE#IuL19=VIB$~_"
    "Vl(Dj*c8^~Kn55tPT7fb5d5mIkKZ7ZlP0>-WT&&{#{~T(1^HQ8c;X^dILr0(>UZp_AP$hTKDyN{Fc3EOhfi=Ga0Vr}eyXV0t"
    "aM$8N*<i_PMbBz#E6nR28UkOk3E&-tMALi-|3phZr9%SLM4FHCFAn!uI_&EGUY`e07il<KE$6GHSfc_kA4hZUeuD5&aB>#kX"
    "&=Uz6Y#tyLp|E<Vz!!Vu+l^=io$XD0(M=j<{U&2faZEg3=Jfg>mk38R0Tr)jNhPuDoY4aSYn#dY67e;$B;t=-@fk=QZ@Lili"
    "4kgE|&gww*$KDbN{izNBAxrr=9l{>c~1@c#ns=l|F`FIQ1T(T5dblCZA{%$Wa*pNU>?!l`V)L@ESJ~UpjD&K0-|1l7;&$g!?"
    "l|DAw>j#E6drzsuF^U<o^i+#KRBqO1sFNq?>Qr^69QC<%%;uqGIT{M^?p5>bGGg%Y+>aB0e-2Q}iy4FN0&|6a|GPe{&<YmE-"
    "v`8P*ruSgMJLS=UPie-j!F7jhIUERlQG33-&)cH7G%%&8!1&v@9<QH&8=MUYDX9hxRbbma#4zEX-Z%)pRPAQ7<=HhzT(a|ye"
    "aB=kG=yKQrfW#Lfc6W33{ml<QjG)Sg9iy{G!SWyR@T>E)>pxzeT#vs0^Y!TX{MCrRqy6|=_7uK4xjOp(RJO&xFMBNX1V&vdh"
    "lVNOD+xfPFi8(sm~AW<0gE*y)Pffj$8yN96)u#0l>ytq1%I!FElH8o_^*|C>;I7AsmvStX$h1!taSWp^oO&X)6<<>L<3MVLo"
    "qSl9yCq0!K>?6=Qr0+Xb)r#r|c7?2exxIt;n<bX9`0c#w-$UE-edGk5hX@fLJw#Is8624DhaKIS)6+0{?PO=6G{;VRd*98?O"
    "*I$+v#5ROw;kB&$Z~SQ?)cXbDh@!|ddarNa1g$?P?CrMPLs)sZ37-Frq$P?ZT|6#5~%5tC}7Fc4WHa$&zVmJJg9eps$Z^FY>"
    "g5=8dL%hA=<rc#n5Rhn}s^rrl_uNl2<e1xe9*K62Ju5ItK&j8iJ%N};VM<W>7ik$0Q6V{k`>2?8T>T_3=9P0@Ws_{K2g$maH"
    "sdDND-8Pt(YZ7`YpD;G>4$OL^T*<%fz|bv2f>s=TDHa!aguX7`k$x#h=8__P#W1~fq}_mpu^be$%ZDB5a#SYy8=xh^^xB-<X"
    "u?l$q@vsDEGMLd-Xxu5$dUTyfutYD03<Qrp&1m7<<WG!<i7Z@<|<U3nDUwb81gaKIZa~EVw%O{Mck~nw@$}8uW@P$err<#WV"
    "qK`A*K_NLEZp`5Ss~Jn~jK2Rlwf%*#Nx?CL|y?B_<^&Ps2m-LjVG=ULWma1j1KWLClTFVP))fdHmtz?Bwco6G$j{ZMeyN>*B"
    "O7yKQ8y5u$;oi8d6P6UewVHU-2eP$C~B_~MIb>zksjZ{9zCr)9KNUt!HL&nPG{zV?@si>mZiT}YuA5;k9I-$ywc@jod521}0"
    "b4>vf9#6|Ft<l=`pK*i^zHwZ&=lfsY|rHrZ2PrbUg((1)4m>@%hWXO1Ma~B&|gC`Y#pPYyJZMSG*3>_=>kOx;MKN6VRRBP(t"
    "PtkbUl|IJMS<_U4Xx$6(?CEUwVKwI)0R1Y+xfXAaKF`M>SZ{NP97&ZM?E^|FAsyV%aIe|&e07pPot&QbcJKHpqu+&E727VuG"
    "d)e3=_Z2@_qNgEH!d(AKgaOR8tDn@zD+SnK9@Z;shTP~oL@;EAOUV5*gVSf7uM1{$0TmCKHtI{gx#XugrNl10tXE{__<T}=B"
    "u82Jt{^qfVc9E&i?*%wm*+<*+P7pKvylI$v%b3@;OU#{`E06P8&9~k<r)qKsFTYT1-Ls0b&bC4MzUA*dzk?tnzMULTaSZC(x"
    "1R+!4=+!ObbPXA;39_-OENeiq*Y2`!@J150@#S5Dn*IBdKVF&as;PqW1bxCJbfU8ulGzM3q>P9m3pCA;TAi0IQGpHFB=?{wv"
    "|;d_0r#PehAKY2iczf)l#`iw{U*Kfz~%kQ<{4P{WGVtDDXI`heF4ptLO*$?ek+3h<dH-LCpM!vnurz!Db)uw!wAzC@aw}&R2"
    "IR2M)_`ZUIiXFW&o?D4V^4+q}59Q8T=koZ~O=rvQ%_SqFbC5D1Rm#P{G_%P#S@NtiD6PCmq?*q7&dp+#@O`!-2}_u&`f$Voz"
    "NIfo7(YMMc%T2Mb_MdB!I>jND<`IC6^g_$75+<>t7xJa5@XuuNmqjR^1IomeN1Np&;qV&;4^BUHuvv3q$n5TdAY^HHy%Xuy@"
    "A{k?DAa{YN7n5Ib^7OCTViVD&<hFx5{%XLxT6SB`c>vpJ)!ZY`MQ$(AYa7<*{)V@~4d{$X4Y5zHCd(=zleBW5Bgin;%3v>%Y"
    "r44>#Y`I(dF4n*Z-@T|BnLVEG&sJ&Bkq2UJDE%XFJEUr^CkCL2onicb|Dkx2xN#FY<k5LY${eputow*Bb(;zJ=)gjMFks;%b"
    ")=5|R_6M2yk-WjI4P>+@2C_ax#E|e;t$*OVmt0_<HOhX=|D`(L*Nb$4`;gTI4b5Z9quSHr#4UqJ&MYK0%@b&2ERpYoQRkcD>"
    "A@nMVRt>M*+}<_0u7lqw|Ku%(F<}7So%eg|{%qMQ+EHDK^rz|ul%jO$6k(}F+v=slduyiVbu~e_x(Tk{SNo>lX{lKL`{)BJ)"
    ";xj!7}KN_m3aaPW>MKXE4K*MS`EcxKtJLONdVe2mOqrz0Yrp3ap?oc$)9CP*c=Ll|9o+Bww|Ct_Dwd=sS=p#+b?K<tJC+~)="
    ">Vh8rh?Kv~9bqLZ30;?}8ws36%jpo|5B9t0sw7&1euo`LeIW#y|$QlC{;Ro%~7kK=@TaMltrf!(C$l^_)tEE**GDf%PYORH@"
    "1#+a@l@qw$y2M1>)(53rUup=Q(<OQ-;xSVu}~uVogE>&BHb4f`YqWI*J4H?Pp8N?w$KnK=cO<yvSbOE^Aomx+sGk?6Rru8`P"
    "FEL0#~t+Sg*C-1x;a3EIIPjUsy6%-@pyPdsWRsU{7Gqs@~NdG-bKT^~(m6;Z`g#JSMHaAzNvC6!xrhHA~Hqn&p^tPM4?EoWB"
    "Lh|i9v`5q#{B1_NnFHAQt|W!p62rX5IZvO*5Z4~*QC>{GTywJ61QT9wtJCTAO58V%#SlR0_eCn)lErX_1*eGpZra74h!Yo+*"
    ")lIVXMO5K{>pdiy%rfilU-@#Ngdoriw`NC*^o>mqJDo{<x3u1u4CbL>hkGOPGQnGO&WEb?%I&@n0dfxuLA8wJ2Kr&Qn5_6<s"
    "scp5hE$P;;*&LcT~Xph|;l&gx_gF2YOjeQB0UmWwzEn|2X3x#lYmVf>+*G6Axz92gHnfe8wWeRJJ6W_}WtC%Q4^io-Uyw=)i"
    "<<_1Bh%gmiV^<yqX+wipD&C&CcFJoYxbCC=f@^;Nheb{Vx9<YCG;J}!7pxN1oap^~oADWq6Yx}#+G<2a{e=s3#9HR)A*){QJ"
    "iIhOdX+bNnuA23i8E?}h>%t)t+#A*Rf`L{*9Uww!cWAu`*f$0uyT&V*zVK!chmsj^Ik--?|E4Wq|e<h+py*GZl9{Q*&qAwo~"
    "wfyY|VWW96BcCA_k$=b^`42qW$>{<uy<DXnOen1ZJ4BmLlHe&zCXsOac2AQj?JC%_(gHVfN!7L<u7_B8`dE{Bt~E2Bk_c?aX"
    "h+JLrQ<AP3rHFrrxCsy?NJmb(Rj|^Y%20PD8cgc_@1CTA1hX@21fu)iFmcq@YLz*Fl5AOHle)>`t@yRmSwP;WE$k>N!oc|wS"
    "+|0xBSIJwR=gR1JCt1Si>LfR`9x>%)QTtNqz-}ET>-$J;){GU5eh`6As}61Fv@8%OuXX>mxa=VQzY)%s~F?qHS6GEmPrd$wO"
    "_c`}NsGBJDP3-Hm*&N+KdDX&!H4UH)&$$^7qa`AsrTSe_OsVuhj^rFsa)(n$-R>Q;{x)`Ngw2y7g#p&sEqDb%AN6&KeT<${P"
    "-dyld+P70pwns=2ZZR%2{q}@?6Bju!X5G|kI0gF{)P6CUGR@!;6mM$t_7a4ng&Tz~FVQgNF995QvPL-_+>Q$GhlIL7KsiRJd"
    "rlGJ0f*j%iR3)ETWk{<adW)i%mf7B7Bs|hjK73T&(uJd*M~QG_^5Vd1nIa~SB1q1xrkz&DLQO;AXW963w?iZ5S=#a?zQZ8T("
    "GC7Go)b;0DV?29FagEyCl}$X(GRC!&R0B-gm5s-Njb|ZFLC0oZ77Q8MNTROngx`v;Bgu5mb|&w`eG-a!YbX(3TL}2Ch(May0"
    "TcL!Ux+v*X<Lk@^oV#n+#rIay5KDQ8BkR24Clu?*)I*oBPBolD(yqmqqBka9^9Vax1*ht7;*^d5ycql~?+j=~NEK^;GKgnto"
    "Hl#U@#}tb|>}0i5F*kK98d)Dh0XHU4!R<yCl34bX4J2@f7k?NXa-WaTSK-w%ZQc}WDzyDj@UoU?ccwsWw&pOeSaIfyA{w71!"
    "}$maVMdmap0`=%AiL9n}DvHR#99W`s{91KGTJb$&08#^UmU@@D7(Bi}&Z1HPNdfGf`b+)6e*CKuYTFKKZ6(Q+zg^@N$FYK5)"
    "Y>;m5U2;1|CIyl<>GWh0>6YhcfE3;oLk*X+wHBIpjd!Nk+MW;Z?ttdpNH76VLZPu=kZ9s?U`aNbUesyZ_!Uq0wYZaw7x}V%h"
    "q!!DcrOi4RtPns<NHDbiWhseH#ey<jiw+)!p8IE6Bzdo6s>B@7La!AOGmh>okMk`5q))fkBy_4g=JoU{sIz2-Y-zfPbWtNo;"
    "rW&kp6eeZFO=LAEt0@mgA(6E@D3()}v`Km2sTSmM7D$>Y&)i-|V_sQv;`mvuVOYN+9M%F`D{Qdpq0`QjP@qlAX^(;~n7!{Ds"
    "f-cYQ0C!%Ci;U32{elvYg=CDxn_@qp~9HBV4_aEI`pm7{j38MHGuq0H{ikQ{Gvj0nFDwa(-F$Gxpgt9<BkvSJ3kV%ELg8jw#"
    "nz2CM33&s=a!qt?YkOVDgc1=sMovLb4TB%yD6+c^|NVrCur!jjGd_d<W=Lk9RCTjs-CM-4&zQWsrtmMb4O(gC&sNgha5SYDP"
    "UlyYjuNi)`5|E)+8LgvIElnd-VO72eBCg4~uncM*%7`XOxrqGY8>|eE1+4`!4>fUqqzEGN<yc#gB3AiAT^hS(aaYImr540Q<"
    "LNjD86H9zRutjIL+uMi9r2&X$H$Fn)Pvhn7pl4S3J;`+TT`C!wuO?r6~xA2Jekop?l0Hfje~p|)lJeiy|2;uSxp<F>8}um2Z"
    "gthWETJJpT<jDxk$vpH(BKc)(mqjc}jfHo2XU|LWkqqDO}QAmS+iNX@CX_a@fRasG_ladPPhcgz;<P=0i;!Z>j~IN!a_UI7q"
    "3g0o>LJB6gu`A@RzoK|Y$bWIg^yC0P7cfB$x^CvsUkac8>@tp({NIc0x+t?5VJaj*APT7O_{o`-n?g8xDaSIEiHz&HVp3-8%"
    "-F{UF~18bAP6g^*{b$he*IzoGSI*NK1A-Z84C5Cj$9xh0$(b|NJD_)WTGF_{_^A&pvnANR4uQL7m9h$|L&Iu_AD&9l?<@I~y"
    "A10Jy(@c1wIAwA<AZgTB#ctdeQQvL~lJQf!+_>`?NfFbj#FE-n(>TNHv=4W7em(mB{QRfz{O0=N<~lsT4Bw2dk6s;JABD%Sk"
    "IsG^g;a&C^A+$Y$43-U9F!jzekk)^XzS<iE5?C#EXuD9#`6cb)Vm(1sx}*h^TinYU1<4xb`L1^0__%sk3vd(;0Z;dEwye%!e"
    "=D}^?6BQsLR!K$`)O`7E?0I!_e#vD0#McP#!g<XHTKK*n+Dt`k4Q)nBA*TR<{Uq6$J{oYbfQE@9aT4UXQM>ufmI?tE<tgj!3"
    "PekqtgY`BhO=Pv&3C5+d|-{$V_y)6OsL%*#f8I666{^AhXi>hRM*AtphGKE?a|D@+v(O;N+uf-4PznDPPjEyvb81-t&c_-+#"
    "zbmYkDPs=Z<u{$=Ivqc@sb0?eo`XP^?hGCCHuzGy_vA!dsN@l4D<67WX(z;hkSma;4Rg~v@T6N;LrCV02Q7#xiZAE{IDu4Be"
    "Ymsis``X*vE6Zku*P}o2x?7#jZ?tM+PBNQQauqGFsK$jKCRzbaij1Yby=zq}F{zd<98Z(U3Qn2KF_GCQW3tqAm{N2iiWkwzo"
    "@$POs>d|oO<%D&^bGBX8+I+Hj0zr(7d%q{dRi8xN%`S4SdXPlq^DF~QHbv#napyq&R(JfKqaY)dSRO205NBAP>qs&(isyOn9"
    "?oG7PFt(RQqebNDlt0-D6cJ2eR7P9zLxm3@VC$4X<qx!Gxhrv3o<CDMUrGh7Z+EK-wFNK2b%ch*x(M*%M_wx?GT=vJgjqyeg"
    "?$sUdkUau!5FNkQ5&e`z4sYr*!{^7DuS7refoOvZ{9tlZ6^fY{CT5Bt8@B&Ac3i@m?@RIKgbuYDPDyJO=;nZ#NvR3tf#yPa5"
    ";y}v%AmfK$k8qF;dY1xt`<&?ykI!tMgWSVLa^m*k}mdXSu$?ya1bxCAS*ZGVJtnx~!YbVN5M#sr{^?*CJSIkQgfEQTcggjKa"
    "j->MW)=-%hCMUe4ilS64mA1yC#d4fQ{LJ4?&gzN3l|;1pWF>@4Zrh~gQ-YBXh==7`(h)K3YIJdQd2~J6Q#Jd9peyF#ML>&ke"
    "vl>*4*d&Wq?(}LskGP>q)XV8j`I@B%9B;&kD$?Bg=PiK#o@*~VHtq8*g!j;%^yUe*2)wlClqX3i08b&C+o7R!+Wy65FN7T6^"
    "dvGgmp!8f&&WqsGiYZO;rXT)L=ydh#3UMWX;&3yp%QtcjU>KRzBlOj++imB)cl*wYq1VY4*g=Ij;>zM4YYjeIX$Gay6wY9W!"
    "|hG@|{e@jd1LNUH7=nd{5OB@q2f?ML~0l7WAqLwu^lt4I)#icxOnRzzq|ZbL7`I>Astf6(6T*^E<UU-)WR{t3^~E*W=CN+<^"
    "b(0~i|XHMMfD&I;OD^$F~nx9xr$$TQ_ko1zoxfhWyE@DNQFPSw8{hWyfzlHrH5!HCWE-w@8IE($Y4R-yN#tgb<7{MK7B^NAH"
    "KsI3w7Mv_Xt;<mh(6g^XD}$Q#l-g^Lmrkoma8Ycq6n#DeH8n{oF1CX`m}R+VsRka((RRy>#cE=JJUJuF#UAX0J*x;tWJVOxA"
    "+b9uPj(Eo1&ybz{I05ZJB<ZTiU+50G8yyAYvNaxW_QfL-Q)QgJkve-l){&wc6r|X!A>Ke#zWQJpe(8s<)KVo8rr?RdXL>{+*"
    "d~nZKwZ%Ev9V3&zlY_DfY7L?gomaGceq!4Zn0?*&`8<`y!H*4~s)-kB_~9WDLxy7@;U~g#n064<aUSl#alO${p%hLuWrJ&li"
    "j61z+{B;PiqN+#SWkfiCk#YjK*g7F9LJ)vax@m*g!Zfk!PUmZFm1M0M$nV+#EuMZY}BAe!FD#pYkrAqYHNOUIVU6rvz%0gTJ"
    "!qJ}gTQTlQyOKp{p*0b1o*;<~KS;zJ~tDzO%w`cMk$t)}4=2|g$$!sLjJY7^N+DdB7cV#8_aqYNUeW2>_pE$jAIXZgv=kWSG"
    "q_gKI*Tfg%FHxhWb8#aNqIWJXL#nn)f4&)A{wThDb98bRRx8>5VRU)&!wG(TT;K30Q>P?P@I%!_X3AbeeTs`q0VhO>+?y-jC"
    "(r04=seWL2g>Rk>h%VsJHmTvNm1m^cb&+vH-sOQ-Q~(e(KP39SLQD+*_`Jr7M8wViwXqiT(~{)UnpU5;E9%X$`K;fxhKp1Fq"
    "@|Jvi)KWwQ8Z(`}M<1#698&R>JF;$P=$nJ^NCI#Tin0U}#K58L$Yv_I@hYLLDILywhoe>8q*4;$>+ME0#+_t781+p?@`M^~z"
    "8kjFq{6%1{lprlT+GM{ypD{7102Y?2Nit7h6JK)~vD%1c-%d@f!Vsd7Xsagsph0=6{YTQ5ghq$CzkY#ymupF=rDo-{VBpG>J"
    "rvgqb5Jyw)v_|mwsCZ6fHpRRe_cDAF02y2;zO=~*Ns|Pyj>0+{O3mw&XYEVrx^ex*j-6sdrc5km04v=B1uGXd{EN~GgM!(^B"
    "oH`&?KEr9$s~(&Xwz@2?+IAK~nt>wKlUJ65&f+<sITTb;5U)57CvT5{4^X)eqR5>T=gLX+<|2=kmTvQ4WJQU9$(Q1&W}!dIW"
    "8=J1#-pm?C7!Fwi0ZE*PkUAKTa6bAb8vF@W2d!RTMUK@`7y{tu7cO=z&#}brw%(Ft{;9}Z?u;Gcqk>buWb<Kup|MW0GrF5qq"
    "X*whag@Hm$MaD<%gP1XUfPw;5ILmca?~r9$V583-p&i<byJ9pim&;$@<*OL)blWmX}SkeKNS!iFc}YGc03+!ZMST$oh(v6Q-"
    "4fNttyl6>G8b%FhrvoRd`SqtNNg>b6Q<%33S(^FtOQHY&R&u+ZLOlZYPd-Z&LtT1KKy4s_&#G3kMF!&`|w(%#5GJD>9M*Q<q"
    "AF+(JmOtyB*rgY|?bnGlyk%;DMG?<9+R*O3A>WHJP`!RS)gVVA#BP|eZzPvfQK7KvA3P(qmrzfM!Djw`K)=}EbaPEKoFljXt"
    "S|UgxLPgO<fZ5~ovt!x`2uFW9xw?j`2>*C~`P22~Xw>O#5fF(t)w5yQP^VvDZFv%1<X5Gl%4KDC1r`je@k3Fc!*^(c((wt4r"
    "Io+h<sD3EsGR*G3m_lL@+mTc6^ejRE-L=7rDm7$-f~y1NSWEp%q~I(!`9UE^<xS}#@Eg#gEr3Qig2k4zHp=`@vev{%VTD|y>"
    "(nsC4}VM+VviE6^43)S3g*OUUuqb*oRten$^~N8)exy9m6e1xbVu#(3J|nseF1qo#rm3aEP+@1)2ADke$t(x9ZukN)uAi4x%"
    "*ewhW@qWJ<JD_(AenNA=W}4fxe<%Z}<uwp$&|4sWX~G(ri;$&T_<_z|h(inGvkTI~I!m@nPbL;*~wusngsQ@x}}UGOuQKNRO"
    "$N~0s+?0x-sOOXm#_cR3#fvq61lUE*83ZZfEt;+Qj2d-4<r1~JF>Rbmfdm9U`1}(rC7s60P!-wj))7x*?R7PL*^1jy;N4h5@"
    "wpa<Dle_rB;mpRLi&{$}L4m$vwbPjbT0n5ta~JyBx7E>6Xo}}VP_Q_Z7t?VYR0gDe!##G|{#09a0{-%{zho<U1QhEs5v38=_"
    "G8Xw8f~@pWm{B_V|=?R_RZvWoH!6sSn_@u$>p^asH}4(k}Vl?@Kv&j0e2c^Ewe+UTPH8ZDkr_uT3}xMT&;v<iqZK&c`C&xs7"
    "l^TrR8k4klPM?SEsHLs^>=Oi^jSQmEc#wuf^llauDzV=A<zR=~wI~!F_?b%KNa2TfJ)fr!1^-NWqqdO?kD0S1&Je*j9Fg2qW"
    "S8+RQiG7GS*GgO%!t06D)chW7jmQxKA!Sd0nEp+#lqi$^yH4))~9br;1YfqcItfg7SW&%Z0T=7g^<wpSW<f{NYrLFchioorl"
    "pRb&*%HR|=Yy~i)N>T7G;{%Zy33VkH&JAY~DwA$X|SKuBEAr#k8hC+u>*s}H#)|bb&xKs(wT=P11iBt0>+5#ZyR)@l0q4EvN"
    ")u>>t_1%?ruVaTeMBP`SRG7o{YNDe*jgD{Vb_S}e_rv+=$vNmex*A>nK~xsKqbq8LZok~x3dPn|SU}lIt>=0X@-H2scrK2vU"
    "(537rAMZ83{^apcU|OcE=z1y9)oIXMhcztHy7dA&71K0HN~f|NC>#tQvmw=)1#}#AK(0Ry7~QM>-L8-`EnbR486+X6!|5g{D"
    "r((a%b0Sb#G<O!u|N4n<Ki>O(oiFb3(}8XLKEs1mh*82BNgrfqJ4G^us-!oRYNwWI>~1yR~3-id-((AcNNm@3gz<2wU<l|6N"
    "y6LF|3ku4DV%JJ6(u*{g1Qy;_K>=%_7wXf+XeTSD{58&8|Ii1$@fYC8&gt)NEjz;C4<K=ICa+BB3JPFBBHiso%{gn9^Ez`<1"
    "U(ts|kA?w(}1OI=l^O>>EmL0t6UQrP-Q>0$;UE<n)feiig-@YFBKNjf<q9|MnfL^$7s^pz2Qk0iFYU!3efsKBZoK(+%Z;pu{"
    "E^G6;`xTd0?E6TIDJDK_*te@}rwuo^AR1Uf>X4ACysCIcRVu?ms)og1yc>b_R?B3_!>-$O+IrHW8jmo?gXwC*Po7DlP7AmIT"
    "E`SW7Bs9BWLT&1dIz$E9(ny#ru*HrzFy_)E!DW!+q<|qJq<5M|8q0Cx~6M6`<3gyI=@l*jHN43q2Ie{^(<XyUsLy)%KRb9hL"
    "cHgi+U-ozQ_&iujTKRDJZq;Bt`a=tlv|J^<b^@OP>7Irh(Nyq)4-p8;sGUGBVP~n{A3a{lqQdMOaJJmP^4hOLcc-#4BK|S87"
    "tlhSr)r7CxRN&c5`O%N$IVY*}PqYGt~mDj&fC3{A4`yvb+`m})(Ja(xiRHmlh3ZA^TVtqJ7K^A97Qb3<MY=iEvPQ#G?{y|2@"
    "mNPJzpI-M^@WsOS9E=0DtA-+fvqebGiG#<G2QWv0~X!leA3>bFCUzIPyvWorY!5XX5?c>p>gB1OJlnHoHbVbhTCpNyl6E~Jt"
    "ZZ4^uyyXd|_YsxwVf^j^p7|xyb<=@X?@A`1m8@1xfuc*|c<Px*0@v;UvuS<#aTCG@JdX{HHpsHX!A+`?lqBV7SGFF6=AesH8"
    "nsFNsy&fdP~NNfAXQ(ryiIHJ$V()Sv=uv)!m?QmMln?_VI);|bCPbLr*12iF(u5a47F}CI{e1RrpywxEnP3g_0LE(S+5jY3%"
    "V&1a23ikF8CdmG?#)DJeh&tWVy=k$~=W#l2;m@THgVbPliv;Y1#?)(1*s+eZ3f5CE8ddWvKeJDqB#bH&pb|dL_zAMVMOu&wx"
    "A)QpR`OX!B&flvFW`VPo%=L=(QN{kgon={XrQGE+uG6E=3bKR?!rAIe^5r96sNQbTUr)u#d^^ww%4$%b223S#b-_(R{4HHH!"
    "xzPj9;T76QgE_wd-ipJ~XDoUC!c!_g0y0U)mW(L7Nf3=0a@=BUIh0B_PVksYYC{?SjG^y|3sJzOe3_O3P=!>wUQBNo%o1{t~"
    "^`({=7VXG4>#xX>7KPcn4Q=#l^}?!k!pZ?RcU^0}u`+MroYl+4$CDYA{jA)gZFJ)f|5jatQk2~7Z36P&X`2^rSBOo6Eo-A2y"
    "6)OMID@}D$LTyulYuGFAtbcN%jEHbQuu;K(Z17m!plhtL@D<O1@sa*lk2tdx5$MGkJp**(;d~+nnzH;;(@QWLkef3nS6DBA1"
    "xk=>X^58VUdKhc0YQ4N82j{be!CEJGXbD7OJ{VMx|xX*%TgsP3cJu)6{e=DkTqm4GZhK{)jjj%R;J$PtU<3UmRVI!kf#}N+_"
    "ktwX5c7^0v9n4q>}tr@y~E|Kru@QY09w4u+M+`x1H%%W>dSH4k(ng|jBbvW}v>+Jl#!pYt0%NICMwq0VIyez8zo*AOwq+EJ="
    "dM-^#qj+Edjo=c=MhxvQ*w4E$V)5pB}^w1^C&gt?@zI`yIeSDcUz=vg3@>AzLG*>wQzHzq}$seBTX}{Nuc5!&Knk>iji9Ej`"
    "96Ii?@vNYA1HP6JJYVGNw;uA#%ksDS`}M+Y^Tpdvb&4FK)|YL2>9!(G?c!1-#uf+O_|UlzLwJktl5&TnvwlUhfK#+#CFdtET"
    "idJFX|9g~E7Qm!BJ#7zBXx$@T>REs8us^9+1l%qYpPU9jX_CrV<&U0S_2Bb_pPOI#6ZK^tSPgqerv+lX!NB+vVBP8c-*QWB<"
    "Oy!*q3O%S)}_~bxgy?alYk)l+v%RCnDzQYh{tBzV;J)5apsaZv6U<cBD>KWvdD2Cay~9c3$$mMX-<8!R9C75(-QbKXM`yntg"
    "`yoXZ$&^@_30YsCJKYT9+Sy9eoDzlNRfMJR~k?p!7ZnmQvGFJ_-|N&%j6qeQlwbe*{FxT1U34h2_Xyxc49M#+b|DZ>|A6XmI"
    "YxqC0;w6g5{dNip|k4{CLV}D;E1$CnZp_xIao?}{b;GcMgq~iWc!qzLhs&fEa^<~3!>5h+v#Oor`Q`eJq<+s^wnHIa&Gi5Cb"
    "QQ%pMl#eLlU7~$ae3gkfqht^{nn?Q_o}Tji(<_%*?=`SgakJIk>=Re4(g6?=z2R^3%B3d76+(Pe51|+bFco(QI?bw22KR<7c"
    "cuJqd%NPzmS*L{fOMP&t<8U-pC8*hp`Tm2psXTAB`g=WA3XXXX4JTX&G+gp)`zR!RYmvZbvj}&Dsn@?gx+@JeKQr|T`?Kj$Y"
    "2qj&z;qhl|=lw^$1d34{{M89|ZGhdZem)%Melh$#zY-0R(=d@jajN1{4{~AiPdR#wq{j7J7?9s?lmmmAlEO1P++g`I)c($B3"
    "f{x<vcv=KA$HwD7CZ*>$=6a@AFDYgFKEWNfLfN}A60F^s%+85JGH%MXY;lzBX!9`euP;;H;VJa!;qitLoP7}9s4Oh=b=E-%S"
    "**Um`M!;MVc-U0YW+-cIr%)HGV^v<T%<Ckjj*BdSnS@JY@jhZWNb;@SEMmZUJiZ;^~OWJBGpH$gbn=JS;VW0xtb2q%mLTn2N"
    ")lcG;fuOI1>%J0uDG$GkM){F6ap!`#=!#z+6EnIKm~?Hqe_mt8C>67n^H&!k?$g;?4q=BXF0gGIuVII=))?B^7u$VRZ#k^Dw"
    "1rz6;(tx5qps)PEB@xkV)yxXl{J))eqBwsYny_Kg}8}KJn0G4Re}N|)eP}1Q<fTc*nS0At$CWt+6`d!D@mVX!^$SNTd$g_*&"
    "zRRRz6ZQyyMrlNU&48Fo>EdV)FvjH`^7pKQ6JZ_)lL`+vU}bhH`3aTiDxIdt6-K+1m<K*8=HEjY+_!9g8Vfvs%6@+!a~suYC"
    "uG9#s4CX%%YnWu;$bV2Z9doMX*cbYC+qDv+VhZc%&HJ^$hsKgE=RUJc1))mUzMqhh1wBCWB-b(SNLiASQX&F-zz+uaDV%18-"
    "y$+yi+fr2uMVLS<Qm#IoT^>=Y!jY!Cy@vGNIRmN*f$|!cQYRgn)Ry3teY&o<goTqy9uQ)3o60g-lf<@uz8oSg&2VWk0;(PKz"
    "IR+H#Q@aRnGhjkp;lHYc^jg?N#KSgMm^$gpAQ>N0f1JNsKo~0FvOZaK{CP&%Q{04K(UVu4o_(POUa@3MN6Y$x${fmF&L)%9T"
    ")c7I;?H6P)#-;S1gJuBlD?HVoZN7e*CtZi>-)Kzp5hZkw0;v*d1O=a*}PU6tLoi-CD<!N1AM1+w}?QS=YLL4J+c19Wxog?-a"
    "?m6Rgo7C`ir~)Q4@#uj*rC^n~}s}O4(iWdOxcv9C0D%`~3HY``ybhuM9+18GNq@M5-<GA{g0g5OnT9$ww;;CY}}Hy_Su15uQ"
    "Atmp1P%sw~*X9%yB7g3rYDU~>0*YhYeJv$ty;kD{?WhE&9$<(Kk9rG#ZWY>`x#%2)FG`b?y}3Qe&-Jt8F!`?Zo%6*m$(Oe6<"
    "Bkkc=NGEH=pY&NWQR}*?M8hP@qjQFW!-CC|Tzwb@Beu{%P&OP$4MPKv{PmC_UkvGoE8~^qy*Hb5xRW5GEsXs2i7o)8FB*|8_"
    "U6odf4YRA8<KxlQ6?J$vy1BkQI;B6aemc3h*sQ5h$5Ir_6mtc%VAGW1?G&KnW%|lvbo@}9@}c;`f2T@aB+;Bo%!x`DLZj73t"
    "P(&v{5n9|=cU`t7qUcSl`3EkHI=mjrcEoxtH4yX>Bd8?_l%Vx-~XwWqbd}Yh-RsM2bJ^@lS+eNMe<-pgBY({qe@y!=lFDV#I"
    "sXgA6<p7&(D9l5*wT9-MZnDx<Hg6y;8CN)~bs#X^VzRhsSt^T`E@UKnjjj&HPX>k6uLSJ*8zX7ts>Kh#z?HlaBxAl?sg;cAd"
    "8=_a~|fq_<(?<&(9;TYT$=Z;|Ymp@zC5<|>MUe3S5Y6;a$0NG4R>DtBiV%hFtJYF|CDsem^0YYQPO&`x?7;0Sy#E@tK=EDYC"
    "NWHMB9%Q{;Uaf<|{+9rom)@tYtD!crtP}~hDiVsVAsKaiZ5>@!a^-Y2AW@H(+<>fZ(e|ei@?aQYFpv^e=8uBTE*qe}twiZPH"
    ")k`p+<aWtYI!xRcSjpk$1#?vFFN^Xe*`iNi%i2v+gX*sTuNtT0eqkXE>p_60Md##w<h<R67yxgDKU3^-?(dKDeTuVDhU%v1_"
    "W1xReq!k&s`%zq#sURH%|2CKLm6r5>@SMQ*=BpWiYFE^hP7ND8M3G(`Yh-`$dvb1+wZ^^Z5ck&&f|%%=l6UVPpP0Eud2T_w}"
    "(ciDj;ax;x&;W`_#0!#-yG4vFu8c+pm<k!-uNMHu35ei^?peN6mZdto>xX-|fEWH^k9psatXRZRJ+bdQ#8&!T*wl<>paV1ZE"
    "p0B7cC_zL-&A4RwN6sfc=bVK*&P3A@&dMal3^RiAvfd0=UrwYqN`$!Xi|;hLKuPKJ?L6?T#MvC&>uSgCn;eZA6rd;6W6vm*C"
    ")i(zcd*A==z88x97lyg<aM;eS_sWn7poFc+PeBT&LFAr&}lfw10jSZcwF-s__gXUaOwY|fS>JB?2&d0T56{<9@+&<wm|8Apt"
    "Y^A!gxDi^^no~!2si?AcF$0GQlyZEcoU8E^KI)<ZQMQ#gdD@DIB>BP{ZKk|R*=X45hrG<>=J}$y1@W72zWJW=l|{uodPSaJ{"
    "*L%kRP$c1L@bI0i7Kh;D8c{@sqA0sh4J)gD!MtipMfxt`-JO0!>J!yrn9ic!RE^wsw+re^)_z~{r=|c)#+&arJ*^r3m(;fjT"
    "=NaZ~BzCoXP|=X)eTs-W>fY<i`x^1zWaOxZJJErPx4yEl;)NDgjj%de5cj$Ey~8C(~`HVsdekiC1qXU${ITGT43<s?b<X*O^"
    "u(Co6)7{B%ii(3Njc4Q^qiK&{nu|J#Fa;iBHsY3UE_VK|BI<22IdhuVBV4N^>b8Qv4y@be!G5UX<JSy^$u_0%sjba^3W60zz"
    "aHt###;Ktuqxs^8HiF<yp*?xJe=;nsWCV57GqBRdsPu`qd%Y$yTe=6=c7sqWdxd&BEWWEqJ$S3@e5e4Il1DgEM%tntfc;UlF"
    "4j|R>Qan`L$iTm)J4@v4k>VHrZAo7~&Ho&79z8w(adV0V{^1E!WsP0+MsMx1b1((Jfq^nDcmcak1@C;FfLVuVbnt~Q>!EI?k"
    "Kq?SWC9zs1y<fsUlSFY#_xt&fw5{lqL`zV5oJklSu*Tt%yPKH8a1g4A)*H=EqWlbN(Mz3sH#ayim&Staot1Z3h1?)C+b(J)1"
    "3y*x@L2PwIXuj)#{MsA;|SHHDB&4)U_;bLo81RzpigC&qAo$!;X3Yc5SlC?<!JWJWNt`D5D;vQ>zckJpss=niZ`r<muJu=+)"
    "`T8QLw4iL8>n$ob7g(c+f>Jvdt2@;Yu8{1>Y8#LwPS3F$CQXGs`Jq`R1{<^=xub5yz~ykF62(aA%|jbu(WN%&J;RV}|VAN~V"
    ">E0P^{X(L`#SH|?j*)Q6Er!0{r-s^0AQCR+`zo@YK6JM+D6FmLxvRL7kS1W|AzT1>4xUCtyo0rSP|3mGm<5~P91XTk0yzGA9"
    "I)aCxyPa|=Iwf&S*QOeN;dLsMw26!oVESkY;h~;j)uc@DLPw&Aj%@u<GYng=^l_+Jx?Xfd_hRYmz2QqouXTB-+0Nrznu-XxD"
    "|T-NTYMD!Iidf=jlM$b<Z>I7>xOXlp~A7H<j5J!c&^`}ia%2K_#|BPN<C00g-dyMu&Eko!=^Lo;&4ngcgR)YGbsL`mx>Scub"
    "*dAI5o?Jhf*^7rrY`LpMQJ*+w}UkuYddIw^zT-{?c5)<Vj^lInhGtkkZvb^tpNP%9RlZWb5@mDY1PrTcsg95qV;qf8ErOs<N"
    "T(trI*W)NtiyK#G16oo~S3`=H!+!se5l%vSm2p*+zY%R4sm9E%%KHQC3f!0tqdih$9S0)Y?w0L+GetWFqFETY1twMohb%50^"
    "AA_MxV(;Lu!RJRkYVV^9SE+>PGp6SxR2DD1k8P$~)Kb@W*{}hh?r2V7%`Rx16^H0MqqbqLTs@xD?$z7~AQHXML14Y67w%3*5"
    "x4f=0v7tx?Yne%5QiQQK+m*Vaxz*}cVskcT@o2;5V_h56_DLmvsva~J2Z%3DN7tiH^ZVkZ;zSnTM*W~J0(<-6eN`=&jh3lg6"
    "|Y#R>uNg1u2-{F6aPznyy;5!)p_&g*n?32e6V8tkChWe62e^FoC*ljK*LSsBO^iI!=oM*r4-f=R-n&u&Pzs56T0O!>1$UHR1"
    "jzMdAw{Yi&@zv<x*`ltjv0>2%^e2I3=<lN>_kx7v+a31|csD8HNN~4~uXoKWDLR9jGV-qxeD5qWWSWi|e6@Zm-vCA*Ch<_HH"
    "`5yga`YJy(e<kAG~^w$j!sb;v%a>=~ePkt-n%R8VeFUmTxhmCK-Vk;+8gS|qn{`K<$$yM;QNo$}b%q@pgXL#-BW;gZ0uWaFd"
    "cAMr=t@~y#MSqL27*wy>>lB7z}3Rh>HMAGEG?}>s6r5Dt04^(p3OzdW>^N^cren9uFRh7p^Fm){owH^iC&ifngK`4FiKOSA4"
    "ot*thqPEGS{em}}*`m@a9jw_XIT}{tJVkh={zN>REorxRPSFis&>i5Si6dl?jmQii<~Q0%6^@xl50vyo<Q-4hV#yCw_GJKvQ"
    "6>~WFcsze-N7CeORS$-?0k7Q)wFM9Wvg!vi$9A(jn}%NyOwAAzUy0m`KHgGU#quB@0m_u+i_|{nQ6I!m3f|#c#N^wOk&^gOe"
    "+ga&$Y8GiZk7`T^1Xr6FYJ0M7|ZUNVcr5KL9DH)zq-f$WCI%i=4!cF$z5kEZcC6fEiZgSx#&uX`0Blg(%XC$sK^cyacYSJ}E"
    "Sonwz!OC%<w5FLRpU`=cL6XV+JQdp;7!^D^7D-PjJy*voX51(_bjvE{mnVOS2cF{9WtEX&bZ6hx_?>M`@<$j>s}v*rBIDpk;"
    "UGm@>khR<}{ay`#vUIyaXsqV)t&<)$MnHk5aZ-D-(j|B*_g!z$?1&I^5cqp<Xzi3sIJ1uIQbCC+CY{@JB(1KHMo*tW7oVbn^"
    "8(4<S4~*DJBi)E&lQGjw{n(Fe-w)ClWOtn)LEm=jIZk3Zp559T4^4+MkXG~-q*lsoE3l#<wKL4b_G2r`VwNSDse?*pmL#sP+"
    "aA*ckkgE9&(3VmvD_r`{GhEjd5L0JTw77}h|WOp%y4Xz`Icd3mSeKa%beIY;xx0Y%t|spw*AzxvRJpQ7$XdPACpd<)YIE~98"
    "ZOzAwV6iAjCzFNuXP{ZGj!Il1>7~z<(?!Gfn2$m|J2+%t#VF0|gwza2ycB1J7nj<TE=?+IyVNKFSVNO*qwOYUsX|1iqcPx??"
    "A)>0_ozYPb;y=vjJ_My6$^SUXG?OCCj$Zuu;7A%2#fwyvV6N0&Z80gjWBsEKX*j^SC%VY=%lap37Lq}V`9e#Fv*C4re4sRM%"
    "gEVi)9mLBN=#^|ONgk^ho)$(^mmsaXUm`{>;jt6FmArnmZz&Jh&Og~AT#L-j34;;sEu#B#4+7<-T3iOn*EYi1kNpUBYCs0f>"
    "&RpGyA}=toUYYAzcEVgcNOWeWCPotIkbEyQ(PyA1dSF6wSZv!y;zy=yxAh2p^g(RpY_4OLSY`~tN-Qfdoh)%&19Axw@k~8&G"
    "S7nVpW5*79p6PSQQ|Pq@W5wQ;-p5<$~(|@b2yGH<d&JvMyVbR(~k`|cEKP}#=c8x3S8)8#%bU>(BDkAz+JY%z{gO%eh?&C2F"
    ";$CQ5raHJr?y4%b-6O$0#wf$O1o@Q23BdUk{)Zd^5JZ$aW1|_l(SO9R__A2Mk&U9b(0G7qoC9uf02QO~s~Uz!Z39oCX#aNzX"
    "FZv?1teWI$tLH51=(5*DP87|%&ukO8xEF-qG5`$7x0btsaiLzzfeG7VY{ywr7>>qD&d$Vp>eH;u%0^aT1mi6AJLEA)fOSJ;D"
    "0I-VapL25!`ithOJsBezWZjR_w`Nie=tD9rm+bc{@WI!pqCX}|Df!7T)_A}_63=3m8hG{q~fjB@zLz9|jk|hCjXzap{m^$-X"
    "`neiW(fv@MWM$@jFo215vOzH2g0w^aQs^%*pqIFoZi988$}u^X>WOPW$9sAJ%LQd)rB+t&tGZ8lb$ouI%pNRe=ILhUN6;lKN"
    "wNq^9|i%&Glg_R)fiTiLdZymxi*-?23J}c^8??@(zM<SUG(sW(dD1RqqA2b;n4HTSEEbWm2Ji__Fk$Jqh|(Chvl)@Gt9sPEt"
    "AZ%8H2X*eU|zeY%WC2OJQVPHxBG12-dnfzd;{{OT~__&yUYf)r1(cu>@FsaGqg;u}p9=`USGFQdl`+c?{mdaBMen1IU8|!Lo"
    "FnimIfIy?E=rxKOWt92<@aE`pACu`B>ZfF8Q%MF}7hl%Z?5dgR8?n&{DUQW#((0<T-9)!2)_-iu`fhUvjj!)jp>pb{ZZ5GAM"
    "`6Klip6_mRdI}rmZJuAk{?I;4HxW37Z=D7sxz4$Qo1{5sV*Y-2mV8c&=836|k+l6>|sRe$+y116<yAa$ofaQ$b$oFj%n5*84"
    "x?5O4Hg$SZsr^cdvglj~YBZT;!p1}vSj~kojeti|FNGlpJZStN)jb`&VFh*?TMW;6CUigkj3T$WbAjx>+PMS78f5^+VC0B_w"
    "*xDHYA5jJCJ-#3VyNGgLHa;dH;bVs-3;&>qt+eQuXo-E_~b(Bm_%woXdzc%6wC9iIPd|ueW-LJ1^atI>bB=HXfFIGg{6&rC~"
    "I&obVy@Ib*v{k%9xWGfo(zA66Q!DLIAR9Y?^UERu#%GaRX+;gK%KfBS00>Ezr);`nq8>cC-oeB>7kn>sTdNDNlD|=EP+84bu"
    "IQ&8!$`4{E|pP2Gxs#WD-F+yU3R77H*fyRoxPgeN*<z{F6c39wyY0@L`q8<6=(+!UrJA;3o+V?e2yI&dir2HTav&2cerKWpi"
    "16Wqz}AWi`wjLCpL9H?5@ub6C_;{gU4Y03<A3fvyKx}U{PVz|%*fS5MaD6o2CcWby#szo;dHiin%0zZSO=>Wx;h7Gfuk%s}!"
    "@L-@L9ZEilp}|ZGG&F7728Wt0RNTZ@a#{)ch&*5oxH=Yf3$AxzIMNh&07D>*<pT!6o>|~ocvMirfI)<pecjyJnGC1jmTTq%<"
    "G{1@OrR%HU}+4X6EQpTp#EX!K|pu`Dcm6Dxb&NyCAJPhXEwak=BYISImwSeWZ-(=VO9o}i2r47N^^2Z@xup%@`YCyr4|646%"
    "p_NX2??0GUCKwE!}NkIN2G96f-DBEQk$bqeFB9Fdnp7M*q<hAF==h1?=bpDnM9(nxWbZ!ypXX*x3f25`AHPEVSsro8ht=!?}"
    "nYlYm>|K>`e&))Q+67bt_A`*2K5{0{_i!L3PSUu#fG>W={4it8H@a1A)zg8eiCETO5x>9t^}fvzB6aOw=xu>mt;NHW}CSVPR"
    "#D1lU24_mjwf|L;mh$)JK*j)Vwfexy8q+X$S;WR*)3@D;3f+=udn7oWYYv7Z$3-B4j0vib8!;kXqrvB_uB*~9|R(D1%@@1IK"
    "CTZC(dEXf{1gtV6=h$K(sST?G2LN+QQgYA<hQRaBBJijSL<~P1*e3?3u%>>UwSGV1&$LoRoa26|hsd(w8CpOFdPM3NbAii?i"
    "F!6LV+`K|G)Wy__iV7OMUR8#hT)rz(bTV7u#HmGlPE5`1jl;@d}}B}{10s;7IY-UElA)F>lsWD5SH#Rm*P7>uC^CML7V~O0("
    "-`dB=$DCtk+BDOY4yj2*m0enGdo;cf;a99%JAQIP1P?M;I9#%Qy(&gJ*g`5~>5P1E;4^Q@?)Qx;FHhdCUQL$r3#cpuCw5j~U"
    "<}9x<?t1Ki@khlZR0-@vf~ws0*QPg}NUg6CNy0|%QHFpQ?D&_p#_2SFSWAWIU%Bim>*d}F~m_Mx)@-!d0&rS3Ye2W<<zlo;S"
    ";xR(q-*oTU3=-trEYWKXu%{c>|Gg1TK6+HpuH7(d=fLnMDHo0(76q_(v@bpYz$rPBLpcS0mjN%ub+tjPErXL!*ji4S<4|*ER"
    "62TS$OQcCeij-IwKn)54=EH{e08{AzIXciOkhu<Il~y8Vm2OR{ni)ONT$SE!m~R|`ODLwKV-_$fEFen)6#*kd%M3|@69cKq$"
    "bXM4;H(Hp*@dTpRS23owl;x>SYx_wiV{Q`Wy>^4hxm4s8m8+}oX)Wl26%|^!N$a{O>o6AVmR5%a4pw_U9jN#GsB0G%9=)ES5"
    "Zm3cmWFo9oURqSi=J92aJcm5Sg$u1duJ)$v`F-G7uShM93t?^U$U`q~2^?bY~5rG;|xsfFGG}W)Xq&#P*?&y@)|$_zAHf)Rr"
    "E30dP?UM$_#Ko+^P`=<Uq*-6pAq54YBB6}J>UdjMF5gEfbvVxwQ*fVT!(!i{4F8GHEkP}zy;!qx%~!c?J4F91Q}0Pwb{XRi*"
    "T8|I5SVJ@I*YO*K;a`KYcF{qmiQk_DVP-s~vKQe-Ekl9HF2N}@Gb8O;;re42(e}w=T%+%rKCvogSn6cUcbcE%@z8lae08bG#"
    "E8q<y-yA-L6=j6AV<Np>386mistXZ=0~*BY!X5JIe<>E(v?xd$*uEE=5sb6xn?5k4?ZAtH>tY%)&_N1t8OKcm)^;m~WMwkH;"
    "B^PiodXM;FvEcd29E@o1M>xr@hCRs#IZ{e7}v&X;7JSWJ1Tjq?RwD^t2~*_<yBGqKKO(U@{hN|354lOfa@`Q7pw%c5EFp3eP"
    "~NK#=7spG{=^q+lG_6fuDl-EQ`ra*5f$)YVYOcrku9W&yndm)KsDyfS(q8cO!;F2IB~WL}7W%kZ=eARNCvX_zwxdNhN07)?L"
    "0_r4ONQ6+Hr2QPA770h=-}a9lTKu!oti2k1H=gQbI%SYE?{DBB)PB%FVM0t+i)WV`lgS7Py*=Ye}+V|6md5e3^|Ccrl)5Ebm"
    "P3s+kY+zcog;283wJ4t3(KGe9Sr@Qv(R6^sCX?m7VNIWuai=jP035p=NeiHHcsty){G{cqip_@~<Rq&{M+Vu$BU3+xPm@)HV"
    "KQj~d-3*K<%dl>`=YpVd20#OK2+bcECe|;};XuJ%i-0OkEPDXQWM`^+MUOgk8oV1|C}<zc^F5p91Bn1!v;+7V4%Cnh_a2@=0"
    "HX_?1*)42-glG)mGJ*pdi2Y%u^Cu!`r??y3E2|TcOF;-Q^@$H6x;&Z4<zY;Lkzbq%{=l20BcP^s$F{w%AG0}TQS_))Fqz-Sj"
    "7iSvtT(vF*0>P$_{LeZa~u+aKGTK!sdG{iEIe3$9C+|(2E{nB@(>rLb?F+A=yBr8NBd>!tkW_z`{^ICXBaByHzmZ2_Y-kRL{"
    "l2@7kkLa*pj~5P!(G<pyaC<z_lDJSjMFwEqWSkiv)=U?B(p1cXv86ACu91Gqo#uIp%+MUMbWaL-(5%s2sy`o3p*R^q#WA>>3"
    "E5mXvQc~hW-0LmV~!52}0$N>-RDvw5~bKvwsi<kfg89CuLZJ0VxnSoB5o;t<rGY@OXm<9W4Kp-Nn9DIN%I7Y#)J(fDhgf%ow"
    "(8nU!Z@`McWy*9rwE?qXvE#&tRScLL)6Pp|z_WsH$2>5r0mQhgJQ}6Wfy?3<5yjf!I@&&(gwgOIc1apX;8_=v3B2vF2<k1iO"
    "*qQ1Zyq3o?x(J^Gq1XpU1I=B_z!TETo0Zb-~q4+_!4aD^W9oUhpfg}PXL81@>AGItOH1z!d~vG<&0A2WRw#CI3HPN<bijLh&"
    "I9?onQ}-F=+JE_9=z}`o?|=c5w{oc}S3Ln3=z;1~5vU<D(T@w-dlNIQkx#gaj108Xg6+VRHjt_XCOz0ARv1wxD(m1}JRd{}R"
    "vGwZ~HDIA#n8<HVE}0c~SsECKtPHej^R40sSG?5C@fw*z|*%mWAnUIM#iS>#kK^5(6#^;q!yfyaVl=$6>?z!)~vHf@ptki$x"
    "OS>V7=!5o3FfhU1M5;A8{eJ1olYA0!iez$bD8K@OqMt%^$?*tGAaP+{YiJL%DoB&=iECcKjXqUxaM3G`BZxiq_hBZwy2UtXp"
    "cj$6K$IxVVSz>x5YRo3tf`bG%6umJcwM^K1C-V|mV?rA-yrLKypL95E6!@6ItK8OI8OsrU(R@4?a>3wb=r-V)4Fbc%v~4)P="
    "m}~L4jSA~7z&*{0uzSB&7usHcdT?vUz?F=(IxYNUnn=kF&&C)QWTNQ5ZMf9PCIg-cPM%WpzQb*$2UyVjsn_KAr9NpC1tl4oG"
    "QCZ2@hukkxe&+@8&{p!})-yL&GozmyWp>(04$II5zMVv<qAUJBBBcCc(BD)&sX{dNGCPq1;%U(YCwg#wmzqP?RxCET}qoI?z"
    "PGjJl;KF+`F=E|%##7+$hX9Aq}E$o)~`022Ux?C3yKz@l-$u!=yZb^>Gu00(#lR0sPDM~kICybISim`*4Tpu~&5+MV!iL}#`"
    "EM9%_0OMEb<3sMCH104$p#>28DHf<ee%rXJ{DZKBxh7bFZMp4F0W82a-g{4KOae^MfYbLlZ4YCwgm+TU`=q})+;X<MMHfBk%"
    "HGooscV&}d_9EJEGPiZAV(!wRls#e<Qmi1%PekmVLWM@~B>^HJ)6iPbEgpp_QXsq3gzpH5>|t0b=<fNZx4p-;aJZ7gG}f~O3"
    "<vb$!K3mFfEdbZ;F#DWhIJe`0ub=jfeJKDqJ>FjEVe5Ee6!1qpt$JK1BYg|<JmBF(4i6Kc||@0*E3-580IQ*>_CS;HF?l6f="
    "6!XdKxgxhJ&?jE!RThl6N58Cf_c1A#qudS%w3~1?Y!B!buEZ>UEf28(m?hOi!2<1xe)42C$iI(`qmebwyezZek1gi$1WS>A+"
    "bD0@@0Nafky*sO1=j4;bu45jp1(_|>3b7ce=1xmody`Pnai%bE(tH>sM4c+iL6?ST6oU<T9B!J-fqIN_8|gmoncjW&)<AoK`"
    "`%TD4{$6%m$^)1~ss+trU1MVhe5bH)p9yTx_Tx!csQ=srDO#ta*p5FjK8bvYS1&e|#Nun74pKj}KL-ygWSs5Aw)0l?Md>}sq"
    ";^jC1E~bk)$1uu3-C$eYacu{JVn=b1z+M>?d-#U>dK7tz$%M{ks2V4&nTl_Yu8&`juEOJ+%S(J+JuKdwgt*HKJLMNXqoL;1!"
    "Dlqoo)!3v#@ZA9p4C`;F5fenYf004c1w*Jc+YOFDY@<$jcv=Ddrot$DRIwetR>s+SuM3BvOS}%mK?QRTKY=*+0&b7%q&yQ;J"
    "4G0(PeuxZAoK0k7H{-*zT>gq<THI0q>VK;Q!JFf@e4I#I&j>x4UD;)Xwd0NrrlA8%=pmPi><uh3Tp7v}Gqfy`8qiqo=mim{a"
    "t;MmE!hp4-kwCeTwGYESZca!W1wJ5Oz=B{k=%%`|4+?9#}N2{lh_YL{G@CpNiD8q5=${7Qz)6I<IQdF6>semyVci4E?aa`Lo"
    ">cgQArVuL#*iafE!opL~)*yi)nJD%9dj+q-zY;}jEj3>6ZTfW5;8~u7J#S<IcElc9;{Cf2L`T0-b`OWpk%{2gK_-1r{^y=vP"
    "C=?}v!}GJ#KmUSci0AjPE$88>tv)NYVHb+;mbI{p%dm4o!Y>)tMy|mNH0HcyW9~~f=DlQN{!2C%ykKLM-S`)1%y`Mh%$IEJB"
    "_+UkNeM7sQUZ*ZlmO!;CBX1sP@fsWlN#G?Z+MqE?!N8&%UY}M=svTh_08JnHPf~a`@Ghg_C220Qsb7!Gn;DI82u$JRrfuf*G"
    "~I(<}=%C-l=?cV~v}T&ugt^fAM*ZHEta~v#GXS!Cji$ZNu+*Ej^p@ciE=f)=p&yZZ1kWRd(KX(y;PjJU*VzSJkR1FO`;4j4i"
    "D+o>8v8shf04JvJ##Eyx(1u#N&h4KhE8VjKO_=|RI~USP+Dqo;HpH1Oj}QWF~Iw=kg1x9S_MoD_cm>Xy3vRK6qJ684F{4%fe"
    "*Wm(Ra-|$+6j#K|_0ZA1m?o3qPUsB288Lx_@t)_GxiZ))#YH;$#v+no0?)3)Wlnp(qKi`W#9-nN<+E?wRGMk!L&C8e_@RKcx"
    "XBm3xrM~54k!#t37g(-kMUn5||Gm_V?7+5DI+yJ+D~%nalB)fLRilzeEeqH1Q?YiYX9e#6k+rMfhW2%PsT8rsb+Z!Frjx;`?"
    "zvzE(}=UgF)W{?&WT+wNaD<<!b1=--40^cv{TFT4VI<0YewcX)~$KX#ARo#tLEB{6SS?`B`;lHR&P%tx)_d1-$&yK=v)?K!?"
    "FdsU0=5Tv&PFyZH<%Vkl*>$Hf-B5{8mLV`b?pu@v<EF&*M49x|+%o{yBr{UyhTU58HX#uqVyqp3FXxQA*n71A5eI8}bb#D<0"
    "pjW~*E~e|>rkVOWgmT8VyH9!>}d#yo-{_@|B8^8fwDY})^JWA;7wUo_@lB8CR#pZ#mZ@Rto6FzDAA_7iN?nrGT>s9q|+xY<z"
    "IbnH2@{0L?TCeaEqGc&!!qB2Pyoh<b;=DUGm*i=@+4~)c&Vm(TY%#2wOy|khFoV4nE7;F3A&r}(fZT_oyttEWX&T=o6U*5=a"
    "FW7j&a$l!GgYVfs|CdU5Z^rblqi5u3MQIFU<xt)#{4d9(a%`qyqdPCsbw5h1EH!`*Dg8UPQkU|5t>*dv0+Fj-7S1=KTFsP=W"
    "%|D)fmF=&Ny9hvtsV--e0#T)m1If&s7fwVLJC{UFJyVr!0D2S1n$VPdmrT=dWW)n>pUI2ik8t23v|nW8ALg+JDW{r(X!VYEM"
    "}7l`s#k_-;bx^oGr$)G#|pY=d1f}Z?K$&)7Aa_p-VUOi0ZJ-B&zq{*FKG>>Fm?+(=$fG-+b<HN{_=aTVe3`W4ddGE%rrU>Vg"
    "|F08NzO&x-|t6A5q8w0ciokb4$QyFWLR;O92l(|xS>tNY%7zx~7|5IvUU?`yKTqS>zwW|RC+A5q2cGtFOf6?V5xO_97UhUtY"
    ")1~n??z%v3Ta*fz@;v{7zO9IF9o!GE~$jBTof%|SI86B=PSU`F|aD1v!!xBpe*81mRA=h()|5I-JQ%#9uVom;iO^IW=w)@XE"
    "CAZPCEil*cQ;C~(-SNIcY6$p5q+5`Pf4Un_7+us!A7q>^9*g_OyFd8r9_ABdeRGd#OEmn2PUSU<dSt{N70ZD%dq7(zmFa{^@"
    "jYFSU0>IO6mF1VM0%P4a~heSnx>oSdK#s!lluR(_?)BLzWMLx^q6pZ{^cI_#>P%Nuf5cv%tl^&!N#B9@3iiowo~DyPKh>B;R"
    "PFi3KbgneA`LzQpZ*sNr34(W~v)hD%=VZH#6)^r=x}eT6O^wU1sW55Tx({lfbmS#3o4MM|R-YhL!#ElnxcT`nQG`x@Y}6yQ&"
    "z!*&ghAF2LAbb+%+TY>QF8%-QQ^V;wWsvV7NvM0mExj397g$26GbQ^9|oYIM3(^e)jI-$($|Er*KNFy<S&AGyzvn@M!r0<|`"
    "ODMNjRZW;gLEzssnlJ<prsUz%-3-^MJKSAd=<-cq({V#N;zS;D@T;oq6KugxnmK=r`I^^F-1XxCoLT@(Ja?j|>4kL&Ge_-Jf"
    "03M5~7c!6~ixMAT+=~(*cmsdxRNUCrpCP`BY1$&bo4-^XCEK(A6^_zQiBj7VP+q2FLgV7m)+Civg*Rpx9z3y3cT$_tB{op;H"
    "nXsJ7IPyijtD@PdK9<@R$*pDrR1OED2?d`;HB0Qx+g<=-@Lf@+E5cOPsZ^;{6#lwt|qMeuG7-qyH2k?%i!xd2J!%l>{p92D0"
    "3Mx;}=L5_j+mFk*%9A0VQNt#Z2G5o3>k|Rxj*o8BNn@k?x}fx^H=XCi!a4?<{*B13xXQ_fkbTnxlG*c54_F>7#<UHeCZl7eS"
    "d;1SLcjaU<aOETKB%a78Q^E&w!g!iyQ>xo2?j-_^bJ`61m2I*8sL_7D5~PfDT62V8@ICyV1noL+etzU%xh@4c^w;M=3hfA8k"
    "K!*|tRi~HeTzTZlwm%BAdPhCu(!J+4{M&4$p%`#2WrBIS1&w66M!iILJv)T)&hkAos^n+)4|K|F^YESR^mG#5!v6)|7G-vBx"
    "v~9T%d&{G`)rJl11^?VM4FC|+4{U=1{ifvsrcuC9cU`9Ih8>tLAX*aX@k^Mu3N2oOd&}0gu&(L(|K_Y47{4axR?@ux4t^!V%"
    "7kEwOIA|8m$PLwAxjI_MX@`-3!HDd(hojw>)rIR)7#q{XIgRbE#15v@{54OP+q>wuiLHM#@nS|yS?7t{|EDrG@A"
)


class Stop(RuntimeError):
    """Preserve the current state and return the durable report."""


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe(root, relative):
    text = str(relative)
    p = PurePosixPath(text)
    if (
        not text
        or not p.parts
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in text
        or any(ord(c) < 32 for c in text)
    ):
        raise Stop("Unsafe relative path.")
    dest = Path(root)
    # Do not resolve and silently follow a link in any existing ancestor.
    for ancestor in [dest, *dest.parents]:
        if ancestor.is_symlink():
            raise Stop("Symlinked directory is not an approved destination.")
    for component in p.parts:
        dest /= component
        if dest.is_symlink():
            raise Stop("Symlinked publication path: " + text)
    return dest


def read(path, limit=MAX_FILE):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise Stop("Missing, oversized or non-regular file: " + str(path))
    before = path.stat()
    body = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise Stop("File changed while reading: " + str(path))
    return body


def atomic(path, body):
    path = Path(path)
    safe(path.parent, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def jread(path):
    return json.loads(read(path, MAX_STATE))


def redact(text):
    for pattern in SECRETS:
        text = pattern.sub("[REDACTED_CREDENTIAL]", text)
    return text


def scan(body, name):
    text = body.decode("utf-8")
    if any(pattern.search(text) for pattern in SECRETS):
        raise Stop("Possible credential in proposed public file: " + name)


def event(name, **fields):
    print(json.dumps({"utc": now(), "event": name, **fields}), flush=True)
    RUN_REPORT["updated_utc"] = now()
    RUN_REPORT["last_event"] = name
    atomic(REPORT, encode(RUN_REPORT))


def remain(cap):
    available = min(float(cap), DEADLINE - time.monotonic())
    if available <= 0:
        raise Stop("Overall command budget exhausted. Return the report; no automatic retry.")
    return available


def run(cmd, label, seconds=30, cwd=None, env=None, allowed=(0,), record_output=True):
    """Capture every command; kill its process group on timeout or interruption."""
    seconds = remain(seconds)
    index = len(RUN_REPORT.setdefault("commands", [])) + 1
    logfile = STORE / ("run-" + RUN_REPORT["run_id"]) / f"{index:03d}_{label}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    LOGS.append(logfile)
    entry = {
        "argv": [str(x) for x in cmd],
        "label": label,
        "cwd": str(cwd or REVIEW),
        "limit_seconds": round(seconds, 3),
        "started_utc": now(),
        "log": str(logfile),
    }
    RUN_REPORT["commands"].append(entry)
    event("COMMAND_STARTED", command=label, command_number=index)
    environment = os.environ.copy()
    environment.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_PAGER="cat",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        GH_PROMPT_DISABLED="1",
        GH_PAGER="cat",
    )
    if env:
        environment.update(env)
    process = None
    start = time.monotonic()
    try:
        with logfile.open("wb") as stream:
            process = subprocess.Popen(
                [str(x) for x in cmd],
                cwd=cwd or REVIEW,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            next_beat = start + 15
            while process.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > seconds:
                    raise Stop("Command deadline reached: " + label)
                if logfile.stat().st_size > MAX_LOG:
                    raise Stop("Command log exceeded its bounded allowance: " + label)
                if time.monotonic() >= next_beat:
                    event(
                        "COMMAND_PROGRESS",
                        command=label,
                        elapsed_seconds=round(elapsed, 1),
                        log_bytes=logfile.stat().st_size,
                        explanation="Byte count is output progress, not completed tests.",
                    )
                    next_beat = time.monotonic() + 15
                time.sleep(0.2)
            entry["exit_code"] = process.returncode
        body = read(logfile, MAX_LOG)
        text = body.decode("utf-8", errors="replace")
        # Redact persisted diagnostic logs, not source files or Git objects.
        sanitized = redact(text)
        if sanitized != text:
            atomic(logfile, sanitized.encode())
        entry["elapsed_seconds"] = round(time.monotonic() - start, 3)
        entry["log_sha256"] = sha(read(logfile, MAX_LOG))
        if not record_output:
            atomic(logfile, ("Output intentionally omitted; SHA-256 " + sha(body) + "\n").encode())
        event(
            "COMMAND_FINISHED",
            command=label,
            exit_code=entry["exit_code"],
            elapsed_seconds=entry["elapsed_seconds"],
        )
        if entry["exit_code"] not in allowed:
            raise Stop(
                f"{label} exited {entry['exit_code']}; full diagnostic is in the report ZIP."
            )
        return text
    except BaseException as exc:
        entry["error"] = redact(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        if logfile.exists() and logfile.stat().st_size <= MAX_LOG:
            try:
                content = read(logfile, MAX_LOG).decode("utf-8", errors="replace")
                atomic(logfile, redact(content).encode())
            except (OSError, Stop):
                pass
        entry["elapsed_seconds"] = round(time.monotonic() - start, 3)
        atomic(REPORT, encode(RUN_REPORT))


def git(*args, root=REVIEW, label="git", seconds=30, allowed=(0,)):
    return run(
        ["git", "--no-pager", "--literal-pathspecs", "-C", str(root), *args],
        label,
        seconds,
        cwd=root,
        allowed=allowed,
    )


def status(root=REVIEW):
    raw = git(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        root=root,
        label="worktree_status",
    )
    result = {}
    items = raw.split("\0")
    i = 0
    while i < len(items):
        item = items[i]
        i += 1
        if not item:
            continue
        if len(item) < 4:
            raise Stop("Unexpected status output.")
        xy, path = item[:2], item[3:]
        safe(root, path)
        if any(c in xy for c in "RCDU"):
            raise Stop(
                "Rename/deletion/conflict requires review; no destructive operation: " + path
            )
        result[path] = xy
    return result


def assert_repo(root, expected_branch):
    actual = git("rev-parse", "--show-toplevel", root=root, label="repo_path").strip()
    if Path(actual).resolve() != root.resolve():
        raise Stop("Command is not in the expected checkout.")
    remote = git("remote", "get-url", "origin", root=root, label="origin").strip()
    if remote not in {
        "https://github.com/" + REPO,
        "https://github.com/" + REPO + ".git",
        "git@github.com:" + REPO + ".git",
        "ssh://git@github.com/" + REPO + ".git",
    }:
        raise Stop("Origin is not the original public commodity repository.")
    branch = git("branch", "--show-current", root=root, label="branch").strip()
    if branch != expected_branch:
        raise Stop("Unexpected checked-out branch. Do not create or reset another branch.")
    if git("ls-files", "-u", root=root, label="unmerged_index").strip():
        raise Stop("Unmerged index entries require a conflict review.")
    return git("rev-parse", "HEAD", root=root, label="head").strip()


def manifest():
    directory = Path.home() / "commodity-public-update" / SPEC["release"]
    body = read(directory / "manifest.json", MAX_STATE)
    if sha(body) != SPEC["manifest_sha256"]:
        raise Stop("The reviewed publication manifest changed; do not make another candidate.")
    value = json.loads(body)
    entries = value["entries"]
    if (
        value["repository"] != REPO
        or value["source_pin"] != PIN
        or len(entries) != 92
        or len({e["path"] for e in entries}) != 92
    ):
        raise Stop("Publication identity or 92-file inventory changed.")
    for entry in entries:
        body = read(safe(directory / "overlay", entry["path"]))
        if sha(body) != entry["published_sha256"] or len(body) != entry["bytes"]:
            raise Stop("Publication overlay bytes changed: " + entry["path"])
    return directory, value


def source_gate(value):
    if assert_repo(ROOT, "main") != PIN:
        raise Stop("Research source pin changed. It will not be reset.")
    for name, expected in value["source_hashes"].items():
        if sha(read(safe(ROOT, name))) != expected:
            raise Stop("Research source/report advanced after the reviewed candidate: " + name)


def nb_sources(body):
    nb = json.loads(body)
    return [
        (
            c["cell_type"],
            "".join(c.get("source", []))
            if isinstance(c.get("source"), list)
            else c.get("source", ""),
        )
        for c in nb["cells"]
    ]


def notebook_gate(body, template, plots):
    if nb_sources(body) != nb_sources(template):
        raise Stop("Notebook code/markdown changed, not just its output.")
    nb = json.loads(body)
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if not cells or any(c.get("execution_count") is None for c in cells):
        raise Stop("Expected saved executed notebook cells.")
    outputs = [o for c in cells for o in c.get("outputs", [])]
    if any(o.get("output_type") == "error" for o in outputs):
        raise Stop("Notebook still contains an error output.")
    if sum(PLOT in o.get("data", {}) for o in outputs) != plots:
        raise Stop("Unexpected number of inline Plotly outputs.")
    scan(body, "reviewed notebook")


def visual_basis(body):
    index = json.loads(body)
    keys = (
        "notebook",
        "title",
        "status",
        "evaluation_origins",
        "completed",
        "comparisons",
        "new_training_fits",
        "cumulative_supervised_seconds",
        "supervised_seconds",
        "elapsed_seconds",
    )
    return [{k: s[k] for k in keys if k in s} for s in index["studies"]]


def patched(body, declaration):
    text = body.decode("utf-8")
    if sha(body) == declaration["patched_sha256"]:
        return body
    if sha(body) != declaration["input_sha256"]:
        raise Stop("A source differs from the exact supplied version; no approximate patching.")
    for edit in declaration["edits"]:
        start, old = edit["offset"], edit["old"]
        if text[start : start + len(old)] != old:
            raise Stop("Targeted source-edit anchor differs.")
        text = text[:start] + edit["new"] + text[start + len(old) :]
    body = text.encode()
    if sha(body) != declaration["patched_sha256"]:
        raise Stop("Targeted source-patch checksum differs.")
    ast.parse(text)
    return body


def save_state(state):
    state["updated_utc"] = now()
    atomic(STATE_FILE, encode(state))


def file_map(paths, root=REVIEW):
    return {name: sha(read(safe(root, name))) for name in sorted(paths)}


def check_files(expected, root=REVIEW):
    for path, digest in expected.items():
        if sha(read(safe(root, path))) != digest:
            raise Stop("File changed after its verified step: " + path)


def save_backup(paths, state):
    destination = STORE / "preserved_review_files.zip"
    if destination.exists():
        raise Stop(
            "An existing backup without a completed transaction is preserved. Return report."
        )
    temporary = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in paths:
            archive.writestr(name, read(safe(REVIEW, name)))
        archive.writestr("snapshot.json", encode(state["before_files"]))
    with zipfile.ZipFile(temporary) as archive:
        for name, digest in state["before_files"].items():
            if sha(archive.read(name)) != digest:
                raise Stop("Backup read-back verification failed.")
    os.replace(temporary, destination)
    state["backup_sha256"] = sha(read(destination, 250 * 1024**2))
    save_state(state)


def self_test():
    class Contracts(unittest.TestCase):
        def test_absolute_path(self):
            with self.assertRaises(Stop):
                safe(Path("/tmp"), "/bad")

        def test_parent_path(self):
            with self.assertRaises(Stop):
                safe(Path("/tmp"), "../bad")

        def test_backslash_path(self):
            with self.assertRaises(Stop):
                safe(Path("/tmp"), "a\\b")

        def test_empty_path(self):
            with self.assertRaises(Stop):
                safe(Path("/tmp"), "")

        def test_control_character(self):
            with self.assertRaises(Stop):
                safe(Path("/tmp"), "a\nb")

        def test_symlink(self):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "real").mkdir()
                (root / "link").symlink_to(root / "real")
                with self.assertRaises(Stop):
                    safe(root, "link/file")

        def test_atomic_read(self):
            with tempfile.TemporaryDirectory() as d:
                target = Path(d) / "a"
                atomic(target, b"first")
                atomic(target, b"second")
                self.assertEqual(read(target), b"second")

        def test_nan_receipt(self):
            with self.assertRaises(ValueError):
                encode({"number": float("nan")})

        def test_no_credentials(self):
            with self.assertRaises(Stop):
                scan(("AKIA" + "Z" * 16).encode(), "synthetic")

        def test_redaction(self):
            self.assertNotIn("Z" * 16, redact("AKIA" + "Z" * 16))

        def test_patch_once_and_reuse(self):
            original, desired = b"x=1\n", b"x = 1\n"
            change = {
                "input_sha256": sha(original),
                "patched_sha256": sha(desired),
                "edits": [{"offset": 1, "old": "=", "new": " = "}],
            }
            self.assertEqual(patched(original, change), desired)
            self.assertEqual(patched(desired, change), desired)

        def test_patch_rejects_unrelated_edit(self):
            change = {"input_sha256": sha(b"x=1\n"), "patched_sha256": sha(b"x = 1\n"), "edits": []}
            with self.assertRaises(Stop):
                patched(b"x=2\n", change)

        def test_spec_inventory(self):
            self.assertEqual(len(SPEC["review_hashes"]), 92)
            self.assertEqual(len(SPEC["patches"]), 18)

        def test_notebook_output_only(self):
            template = {
                "cells": [
                    {"cell_type": "code", "source": "x=1", "execution_count": None, "outputs": []}
                ]
            }
            executed = {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "x=1",
                        "execution_count": 1,
                        "outputs": [{"output_type": "display_data", "data": {PLOT: {}}}],
                    }
                ]
            }
            notebook_gate(encode(executed), encode(template), 1)

        def test_notebook_source_edit_blocked(self):
            with self.assertRaises(Stop):
                notebook_gate(
                    encode({"cells": [{"cell_type": "code", "source": "x=2"}]}),
                    encode({"cells": [{"cell_type": "code", "source": "x=1"}]}),
                    1,
                )

        def test_notebook_error_blocked(self):
            obj = {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": "x=1",
                        "execution_count": 1,
                        "outputs": [{"output_type": "error"}],
                    }
                ]
            }
            with self.assertRaises(Stop):
                notebook_gate(encode(obj), encode(obj), 0)

        def test_visual_basis_ignores_report_digest(self):
            first = {
                "studies": [
                    {"notebook": 18, "completed": True, "comparisons": [], "report_sha256": "a"}
                ]
            }
            other = json.loads(json.dumps(first))
            other["studies"][0]["report_sha256"] = "b"
            self.assertEqual(visual_basis(encode(first)), visual_basis(encode(other)))

        def test_visual_basis_keeps_score(self):
            first = {"studies": [{"notebook": 18, "comparisons": [{"official_metric": 0.3}]}]}
            other = {"studies": [{"notebook": 18, "comparisons": [{"official_metric": 0.4}]}]}
            self.assertNotEqual(visual_basis(encode(first)), visual_basis(encode(other)))

        def test_format_selection_is_safe_only(self):
            self.assertNotIn("unsafe", FORMAT_SELECT)
            self.assertEqual(set(FORMAT_SELECT.split(",")), {"I", "F401", "UP017", "UP022"})

        def test_known_output_edits(self):
            outputs = {
                n
                for n, v in SPEC["classifications"].items()
                if v == "NOTEBOOK_OUTPUT_OR_METADATA_CHANGE_ONLY"
            }
            self.assertEqual(
                outputs,
                {
                    "notebooks/18_released_sequence_ablation.ipynb",
                    "notebooks/19_prior_error_memory_ablation.ipynb",
                    "notebooks/portfolio_overview.ipynb",
                },
            )

        def test_original_source_payload_checksum(self):
            self.assertEqual(sha(original_publisher_bytes()), PRIOR_PUBLISHER_SHA)

        def test_capture_correction_preserves_arguments(self):
            body = (
                b"import subprocess\n"
                b"def git(root, *args):\n"
                b"    return subprocess.run([root, *args], stdout=subprocess.PIPE,\n"
                b"                          stderr=subprocess.PIPE, timeout=25, check=False)\n"
            )
            fixed = capture_output_correction(body)
            self.assertIn(b"capture_output=True", fixed)
            self.assertIn(b"timeout=25, check=False", fixed)
            self.assertNotIn(b"stdout=subprocess.PIPE", fixed)
            self.assertEqual(capture_output_correction(fixed), fixed)

        def test_capture_correction_refuses_merged_streams(self):
            body = (
                b"import subprocess\ndef git():\n"
                b"    return subprocess.run([], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)\n"
            )
            with self.assertRaises(Stop):
                capture_output_correction(body)

        def test_capture_correction_refuses_comments(self):
            body = (
                b"import subprocess\ndef git():\n"
                b"    return subprocess.run([], stdout=subprocess.PIPE, # keep this note\n"
                b"                          stderr=subprocess.PIPE, check=False)\n"
            )
            with self.assertRaises(Stop):
                capture_output_correction(body)

        def test_capture_correction_refuses_other_function(self):
            with self.assertRaises(Stop):
                capture_output_correction(b"import subprocess\ndef other():\n    pass\n")

        def test_capture_correction_refuses_two_calls(self):
            body = (
                b"import subprocess\ndef git():\n"
                b"    subprocess.run([], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)\n"
                b"    subprocess.run([], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)\n"
            )
            with self.assertRaises(Stop):
                capture_output_correction(body)

        def test_capture_true_reuse(self):
            body = b"import subprocess\ndef git():\n    return subprocess.run([], capture_output=True, check=False)\n"
            self.assertEqual(capture_output_correction(body), body)

        def test_capture_false_rejected(self):
            body = b"import subprocess\ndef git():\n    return subprocess.run([], capture_output=False, check=False)\n"
            with self.assertRaises(Stop):
                capture_output_correction(body)

        def test_expected_partial_finding_is_narrow(self):
            rows = [
                {"code": "UP022", "filename": "/tmp/candidate/scripts/commodity_public_update.py"}
            ]
            expected_partial_finding(rows, 1, Path("/tmp/candidate"))
            for bad, code in [
                ([], 0),
                (rows, 2),
                (rows + rows, 1),
                ([{"code": "F821", "filename": rows[0]["filename"]}], 1),
            ]:
                with self.subTest(exit_code=code, findings=bad), self.assertRaises(Stop):
                    expected_partial_finding(bad, code, Path("/tmp/candidate"))

        def test_partial_wrong_file_rejected(self):
            with self.assertRaises(Stop):
                expected_partial_finding(
                    [{"code": "UP022", "filename": "/tmp/other.py"}], 1, Path("/tmp/candidate")
                )

        def test_final_lint_does_not_allow_exit_one(self):
            with self.assertRaises(Stop):
                require_no_lint([], 1, "synthetic")
            with self.assertRaises(Stop):
                require_no_lint([{"code": "UP022"}], 0, "synthetic")
            require_no_lint([], 0, "synthetic")

        def test_real_capture_equivalence(self):
            command = [
                sys.executable,
                "-c",
                'import sys; print("out"); print("err", file=sys.stderr); sys.exit(3)',
            ]
            pipe_settings = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
            before = subprocess.run(command, **pipe_settings, timeout=5, check=False)
            after = subprocess.run(command, capture_output=True, timeout=5, check=False)
            self.assertEqual(
                (before.returncode, before.stdout, before.stderr),
                (after.returncode, after.stdout, after.stderr),
            )

        def test_actual_installed_ruff_up022_contract(self):
            body = (
                b"import subprocess\ndef git(root):\n"
                b"    return subprocess.run([root], stdout=subprocess.PIPE,\n"
                b"                          stderr=subprocess.PIPE, timeout=25, check=False)\n"
            )
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "sample.py"
                path.write_bytes(body)
                command = [
                    str(PYTHON),
                    "-m",
                    "ruff",
                    "check",
                    "--isolated",
                    "--select",
                    "UP022",
                    "--no-unsafe-fixes",
                    "--output-format",
                    "json",
                    str(path),
                ]
                before = subprocess.run(command, capture_output=True, timeout=10, check=False)
                self.assertEqual(before.returncode, 1, before.stderr.decode(errors="replace"))
                self.assertEqual([row["code"] for row in json.loads(before.stdout)], ["UP022"])
                path.write_bytes(capture_output_correction(body))
                after = subprocess.run(command, capture_output=True, timeout=10, check=False)
                self.assertEqual(after.returncode, 0, after.stderr.decode(errors="replace"))
                self.assertEqual(json.loads(after.stdout), [])

        def test_pull_request_body_is_exact_ascii_bytes(self):
            self.assertIsInstance(PULL_REQUEST_BODY, bytes)
            self.assertEqual(len(PULL_REQUEST_BODY), 637)
            self.assertTrue(PULL_REQUEST_BODY.isascii())
            self.assertEqual(PULL_REQUEST_BODY.decode("ascii").encode("ascii"), PULL_REQUEST_BODY)
            self.assertTrue(
                PULL_REQUEST_BODY.startswith(b"## Public commodity research publication\n")
            )

        def test_no_unnecessary_ascii_literal_encode_in_publisher(self):
            tree = ast.parse(read(Path(__file__)))
            bad = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                value = node.func.value
                if (
                    node.func.attr == "encode"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value.isascii()
                    and not node.args
                    and not node.keywords
                ):
                    bad.append(node.lineno)
            self.assertEqual(bad, [])

        def test_safe_followup_scope_and_fix_safety(self):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                allowed = {"scripts/commodity_publish.py"}
                row = {
                    "filename": str(root / "scripts/commodity_publish.py"),
                    "fix": {"applicability": "safe"},
                    "code": "UP012",
                }
                self.assertTrue(safe_managed_findings([row], root, allowed))
                for change in [
                    {"filename": str(root / "src/keep.py")},
                    {"filename": str(root / "../outside.py")},
                    {"fix": {"applicability": "unsafe"}},
                    {"fix": {}},
                    {"fix": None},
                ]:
                    with self.subTest(change=change):
                        self.assertFalse(safe_managed_findings([{**row, **change}], root, allowed))
                self.assertFalse(safe_managed_findings([], root, allowed))

        def test_actual_configured_fix_catches_rule_outside_historical_selector(self):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = root / "pyproject.toml"
                config.write_text('[tool.ruff.lint]\nselect = ["UP"]\n', encoding="utf-8")
                path = root / "sample.py"
                path.write_bytes(b'MESSAGE = "publication".encode()\n')
                base = [
                    str(PYTHON),
                    "-m",
                    "ruff",
                    "check",
                    "--no-unsafe-fixes",
                    "--config",
                    str(config),
                    "--output-format",
                    "json",
                ]
                narrow = subprocess.run(
                    [*base, "--select", FORMAT_SELECT, str(path)],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(narrow.returncode, 0, narrow.stderr.decode(errors="replace"))
                broad = subprocess.run(
                    [*base, str(path)], capture_output=True, timeout=10, check=False
                )
                self.assertEqual(broad.returncode, 1, broad.stderr.decode(errors="replace"))
                self.assertEqual([r["code"] for r in json.loads(broad.stdout)], ["UP012"])
                fixed = subprocess.run(
                    [*base, "--fix", str(path)], capture_output=True, timeout=10, check=False
                )
                self.assertEqual(fixed.returncode, 0, fixed.stderr.decode(errors="replace"))
                self.assertEqual(json.loads(fixed.stdout), [])
                self.assertEqual(ast.parse(path.read_bytes()).body[0].value.value, b"publication")

        def test_actual_entire_generated_publisher_passes_configured_candidate_gate(self):
            # Covers the entire deliverable, not only a tiny rule-specific fixture.
            # This creates data files and invokes Ruff; it does not import/run the copy.
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config = root / "pyproject.toml"
                config.write_bytes(read(REVIEW / "pyproject.toml"))
                path = root / "scripts/commodity_publish.py"
                path.parent.mkdir()
                path.write_bytes(read(Path(__file__)))
                before_config = sha(config.read_bytes())
                check = [
                    str(PYTHON),
                    "-m",
                    "ruff",
                    "check",
                    "--no-unsafe-fixes",
                    "--config",
                    str(config),
                    "--output-format",
                    "json",
                ]
                corrected = subprocess.run(
                    [*check, "--fix", str(path)],
                    cwd=root,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                self.assertIn(
                    corrected.returncode, (0, 1), corrected.stderr.decode(errors="replace")
                )
                formatted = subprocess.run(
                    [str(PYTHON), "-m", "ruff", "format", "--config", str(config), str(path)],
                    cwd=root,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(formatted.returncode, 0, formatted.stderr.decode(errors="replace"))
                final = subprocess.run(
                    [*check, str(path)], cwd=root, capture_output=True, timeout=15, check=False
                )
                self.assertEqual(final.returncode, 0, final.stdout.decode(errors="replace"))
                self.assertEqual(json.loads(final.stdout), [])
                self.assertEqual(sha(config.read_bytes()), before_config)

        def test_actual_candidate_normalizer_keeps_protected_sources_and_configuration(self):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "scripts").mkdir()
                (root / "src").mkdir()
                (root / "tests").mkdir()
                config = root / "pyproject.toml"
                config.write_bytes(b'[tool.ruff.lint]\nselect = ["E", "F", "I", "UP", "B"]\n')
                preserved = root / "src/sentinel.py"
                preserved.write_bytes(b"VALUE = 1\n")
                sample = root / "scripts/commodity_example.py"
                sample.write_bytes(b'PUBLIC = "example".encode()\n')
                frozen = {
                    name: sha(read(root / name)) for name in ("pyproject.toml", "src/sentinel.py")
                }
                normalize_publication_candidate(["scripts/commodity_example.py"], root)
                self.assertEqual(ast.parse(sample.read_bytes()).body[0].value.value, b"example")
                self.assertEqual(RUN_REPORT["candidate_gate"]["remaining_findings"], 0)
                check_files(frozen, root=root)
                with self.assertRaises(Stop):
                    normalize_publication_candidate(["src/sentinel.py"], root)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    )
    receipt = {
        "status": "TESTS_PASSED"
        if result.wasSuccessful() and not result.skipped
        else "TESTS_FAILED",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "helper_sha256": sha(read(Path(__file__))),
        "utc": now(),
    }
    atomic(STORE / "self_test.json", encode(receipt))
    if receipt["status"] != "TESTS_PASSED":
        raise Stop("Continuation regression tests failed.")
    return receipt


PROVENANCE_TEXT = """# Publication copies and historical execution identity

The public repository includes the research implementation and recorded notebook evidence.
The September 14 publication repairs formatting and explicit exception/import/closure
contracts in the manual helpers and tests. The UP022 correction replaces only a
reviewed subprocess.run stdout/stderr PIPE pair with capture_output=True. The PR
message is a bytes literal with unchanged contents. Full configured safe fixes
are applied only to the listed manual publication candidates in scratch. No
blanket unsafe fixes, rule suppression, or weakened quality configuration are used. It does not change frozen `src/` model/feature
modules, model parameters, statistical study declarations, raw data, or saved predictions.
The historical AWS research checkout remains at its recorded source pin.

The pre-publication manual Python bytes are preserved as UTF-8 text under
`docs/history/manual_execution_sources/`, with their original relative paths and hashes in
`reports/manual_research/publication_source_map.json`. The active public Python files are
readable, linted review copies. Historical checkpoint gates intentionally depend on the
original executable bytes: do not overwrite expected hashes to reuse old artifacts.
Use the original source pin plus the archived original helper/test bytes in a SEPARATE
reproduction checkout when reproducing those saved runs; required licensed data and private
checkpoint files are not shipped here. Copy only the archive paths recorded in the map,
verify SHA-256 before copying, and keep the active research environment unchanged.

Publication lint/test success is not a new scientific result. The recorded numerical
results belong to their original execution-source identities, not a claim of private-data
reproduction under reformatted copies. Full quality checks and existing publication
verifiers are retained. No lint-ignore rule or expected historical checksum was relaxed.

The original review notebook outputs were preserved where their source matched the
approved overlay. If the aggregate inputs changed, only `portfolio_overview.ipynb` was
refreshed; research notebooks and trained models were not re-executed. Publication reports
link the original evidence hashes to the approved review bytes.
"""


def overview_refresh(template, state):
    path = REVIEW / "notebooks/portfolio_overview.ipynb"
    atomic(path, template)
    command = r"""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path.cwd()
path=root/'notebooks/portfolio_overview.ipynb'
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=45,kernel_name='commodity-manual',
               resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(nb,path)
"""
    run([PYTHON, "-u", "-c", command], "refresh_aggregate_overview_only", 90)
    notebook_gate(read(path), template, 6)
    state["portfolio_refreshed"] = True


def original_publisher_bytes():
    """Read inert archived source for byte comparison; never import or execute it."""
    raw = zlib.decompress(base64.b85decode(ORIGINAL_PUBLISHER_B85))
    if sha(raw) != PRIOR_PUBLISHER_SHA:
        raise Stop("Embedded original publication source failed its checksum.")
    return raw


def capture_output_correction(body):
    """Replace only the reviewed Git wrapper's two PIPE keywords.

    All other syntax, arguments, comments and statements must remain unchanged.
    This does not enable arbitrary unsafe fixes or suppress UP022.
    """
    text = body.decode("utf-8")
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "git"]
    if len(functions) != 1:
        raise Stop("Expected exactly one module-level git wrapper for UP022.")
    calls = [
        n
        for n in ast.walk(functions[0])
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "subprocess"
    ]
    if len(calls) != 1:
        raise Stop("Expected exactly one subprocess.run call in the Git wrapper.")
    call = calls[0]
    keywords = {k.arg: k for k in call.keywords}
    if len(keywords) != len(call.keywords) or None in keywords:
        raise Stop("Ambiguous subprocess keyword arguments; no rewrite.")
    if "capture_output" in keywords:
        value = keywords["capture_output"].value
        if (
            isinstance(value, ast.Constant)
            and value.value is True
            and "stdout" not in keywords
            and "stderr" not in keywords
        ):
            return body
        raise Stop("Existing capture configuration is not the reviewed PIPE pair.")
    for name in ("stdout", "stderr"):
        keyword = keywords.get(name)
        value = keyword.value if keyword else None
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "PIPE"
            and isinstance(value.value, ast.Name)
            and value.value.id == "subprocess"
        ):
            raise Stop("UP022 correction requires both explicit subprocess.PIPE values.")
    lines = body.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def span(node):
        return (
            offsets[node.lineno - 1] + node.col_offset,
            offsets[node.end_lineno - 1] + node.end_col_offset,
        )

    out_start, out_end = span(keywords["stdout"])
    err_start, err_end = span(keywords["stderr"])
    if err_start <= out_end:
        raise Stop("Unexpected keyword order; preserve the original call.")
    # The actual reported wrapper has no comments between these keywords.
    # Refuse a variant where removing syntax might discard a user's comment.
    if b"#" in body[out_start:err_end]:
        raise Stop("Comments inside the capture arguments require manual review.")
    after = err_end
    while after < len(body) and body[after : after + 1] in (b" ", b"\t", b"\r", b"\n"):
        after += 1
    if body[after : after + 1] != b",":
        raise Stop("Expected the preserved environment/timeout keywords after stderr.")
    desired = body
    edits = [(out_start, out_end, b"capture_output=True"), (err_start, after + 1, b"")]
    for start, end, replacement in sorted(edits, reverse=True):
        desired = desired[:start] + replacement + desired[end:]
    expected = copy.deepcopy(tree)
    target_function = next(
        n for n in expected.body if isinstance(n, ast.FunctionDef) and n.name == "git"
    )
    expected_call = next(
        n
        for n in ast.walk(target_function)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "subprocess"
    )
    for keyword in expected_call.keywords:
        if keyword.arg == "stdout":
            keyword.arg = "capture_output"
            keyword.value = ast.Constant(value=True)
    expected_call.keywords = [k for k in expected_call.keywords if k.arg != "stderr"]
    if ast.dump(ast.parse(desired), include_attributes=False) != ast.dump(
        expected, include_attributes=False
    ):
        raise Stop("UP022 rewrite changed syntax outside the two reviewed keywords.")
    return desired


def lint_json(paths, label, cwd, fix=False, selected=None):
    """Record findings even on exit 1; callers must reject nonempty final findings."""
    output = STORE / ("run-" + RUN_REPORT["run_id"]) / (label + ".json")
    args = [
        PYTHON,
        "-m",
        "ruff",
        "check",
        "--no-unsafe-fixes",
        "--output-format",
        "json",
        "--output-file",
        str(output),
    ]
    if fix:
        args.append("--fix")
    if selected:
        args += ["--select", selected]
    args += list(paths)
    run(args, label, 90, cwd=cwd, allowed=(0, 1))
    exit_code = RUN_REPORT["commands"][-1]["exit_code"]
    rows = jread(output)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise Stop("Lint output is not a diagnostic list: " + label)
    counts = {}
    for row in rows:
        code = str(row.get("code"))
        counts[code] = counts.get(code, 0) + 1
    summary = {
        "command": label,
        "exit_code": exit_code,
        "remaining": len(rows),
        "codes": counts,
        "path": str(output),
        "sha256": sha(read(output, MAX_STATE)),
        "is_final_gate": not fix,
    }
    RUN_REPORT.setdefault("lint_diagnostics", []).append(summary)
    event("LINT_FINDINGS_RECORDED", command=label, remaining=len(rows), codes=counts)
    if rows:
        contexts = []
        for row in rows[:60]:
            filename = Path(str(row.get("filename", "")))
            try:
                relative = filename.relative_to(Path(cwd))
                source = read(safe(cwd, str(relative)), MAX_FILE).decode("utf-8").splitlines()
                number = int(row.get("location", {}).get("row", 1))
                contexts.append(
                    {
                        "file": str(relative),
                        "code": row.get("code"),
                        "line": number,
                        "context": redact(
                            "\n".join(
                                source[
                                    max(0, number - 4) : min(
                                        max(
                                            number + 4,
                                            int(row.get("end_location", {}).get("row", number)) + 3,
                                        ),
                                        number + 60,
                                    )
                                ]
                            )
                        ),
                    }
                )
            except (Stop, OSError, ValueError, TypeError):
                contexts.append(
                    {"file": str(filename), "context": "Could not safely read context."}
                )
        atomic(output.with_suffix(".contexts.json"), encode(contexts))
    return rows, exit_code


def require_no_lint(rows, exit_code, label):
    if rows or exit_code != 0:
        raise Stop(
            label + " is not clean; full JSON diagnostics and source context are in the report ZIP."
        )


def safe_managed_findings(rows, scratch, managed):
    """Only an explicitly safe finding in an editable candidate can earn a second pass."""
    if not rows:
        return False
    for row in rows:
        filename = Path(str(row.get("filename", "")))
        if not filename.is_absolute():
            filename = Path(scratch) / filename
        try:
            relative = filename.relative_to(Path(scratch)).as_posix()
            safe(scratch, relative)
        except (ValueError, Stop):
            return False
        if relative not in managed or (row.get("fix") or {}).get("applicability") != "safe":
            return False
    return True


def normalize_publication_candidate(python_paths, scratch):
    """Safe fixes for the ACTUAL configured rules, followed by strict full checks.

    The narrow historical --select list is not used here. Neither src nor the
    lint configuration is writable by these correction commands. At most two
    passes are allowed; pass two requires recorded, safe, in-scope findings and
    evidence that pass one changed candidate bytes.
    """
    managed = set(python_paths)
    if not managed or len(managed) != len(python_paths):
        raise Stop("Candidate Python scope is empty or duplicated.")
    for name in managed:
        safe(scratch, name)
        if not name.endswith(".py") or not name.startswith(("scripts/", "tests/")):
            raise Stop("Candidate correction scope must contain only manual Python files.")
    immutable = {}
    for path in sorted(Path(scratch).rglob("*")):
        if path.is_file():
            name = path.relative_to(scratch).as_posix()
            protected_source = name.endswith(".py") and name.startswith(
                ("src/", "tests/", "scripts/")
            )
            protected_config = name in {
                "pyproject.toml",
                ".python-version",
                "ruff.toml",
                ".ruff.toml",
            }
            # Ruff cache files are not source inputs. The earlier reconstruction
            # has already created a cache that later Ruff commands may refresh.
            if name not in managed and (protected_source or protected_config):
                immutable[name] = sha(read(safe(scratch, name)))
    RUN_REPORT["candidate_snapshot"] = {
        "root": str(scratch),
        "python_paths": sorted(managed),
        "policy": "Full existing configured rules; safe fixes only; no selector override.",
        "maximum_passes": MAX_CANDIDATE_PASSES,
    }
    for number in range(1, MAX_CANDIDATE_PASSES + 1):
        before = file_map(managed, root=scratch)
        rows, exit_code = lint_json(
            sorted(managed), f"configured_safe_fixes_{number}", scratch, fix=True
        )
        # Findings are recorded even if fixing cannot resolve all of them.
        # A separate final check below is authoritative; exit 1 is NOT a pass.
        event(
            "CONFIGURED_SAFE_FIX_PASS",
            pass_number=number,
            maximum_passes=MAX_CANDIDATE_PASSES,
            outstanding=len(rows),
            exit_code=exit_code,
            unsafe_fixes=False,
        )
        run(
            [PYTHON, "-m", "ruff", "format", *sorted(managed)],
            f"format_candidate_{number}",
            90,
            cwd=scratch,
        )
        check_files(immutable, root=scratch)
        rows, exit_code = lint_json(
            ["src", "tests", "scripts"], f"candidate_full_lint_{number}", scratch
        )
        after = file_map(managed, root=scratch)
        if not rows and exit_code == 0:
            run(
                [PYTHON, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
                "candidate_format_check",
                90,
                cwd=scratch,
            )
            check_files(immutable, root=scratch)
            RUN_REPORT["candidate_gate"] = {
                "status": "FULL_CONFIGURED_LINT_AND_FORMAT_PASSED",
                "passes": number,
                "remaining_findings": 0,
                "configuration_unchanged": True,
                "immutable_candidate_files": len(immutable),
                "candidate_python_sha256": after,
            }
            event(
                "FULL_CONFIGURED_CANDIDATE_PASSED",
                passes=number,
                remaining_findings=0,
                python_files=len(managed),
            )
            return
        can_retry = (
            number < MAX_CANDIDATE_PASSES
            and before != after
            and safe_managed_findings(rows, scratch, managed)
        )
        if not can_retry:
            require_no_lint(rows, exit_code, "Full configured candidate lint")
        event(
            "SAFE_FOLLOWUP_PASS_PLANNED",
            remaining=len(rows),
            reason="Only safe, editable findings remain after a changed candidate.",
        )
    raise Stop("Configured lint pass bound reached; review copies have not been installed.")


def expected_partial_finding(rows, exit_code, scratch):
    if exit_code != 1 or len(rows) != 1:
        raise Stop(
            "The previous safe-fix state was not reproduced exactly; review files are unchanged."
        )
    row = rows[0]
    expected = Path(scratch) / "scripts/commodity_public_update.py"
    if row.get("code") != "UP022" or Path(str(row.get("filename"))) != expected:
        raise Stop("An unexpected remaining rule or file prevents known-state recovery.")


def publish_transaction_patch(state):
    """Finish only recorded exact before -> planned byte replacements."""
    planned = state["repair"]["planned_files"]
    before = state["repair"]["before_files"]
    root = safe(STORE, state["repair"]["candidate_relative"])
    managed = set(state["managed_paths"])
    if set(status()) - managed:
        raise Stop("New unmanaged review changes appeared during lint recovery.")
    for name, expected in planned.items():
        data = read(safe(root, name))
        if sha(data) != expected:
            raise Stop("Prepared recovery bytes changed: " + name)
        current = sha(read(safe(REVIEW, name)))
        if current not in {before[name], expected}:
            raise Stop("Review file changed during the recorded recovery: " + name)
    for name, expected in before.items():
        if name not in planned and sha(read(safe(REVIEW, name))) != expected:
            raise Stop("A preserved review file changed during recovery: " + name)
    for name, expected in planned.items():
        if sha(read(safe(REVIEW, name))) != expected:
            atomic(safe(REVIEW, name), read(safe(root, name)))
    state["repair"]["phase"] = "EXACT_PATCH_APPLIED"
    state["repair"]["applied_utc"] = now()
    state["repair"]["installed_files"] = file_map(managed)
    state["steps"].append("EXPLICIT_CAPTURE_CORRECTION_AND_FULL_CONFIGURED_SAFE_LINT")
    save_state(state)
    event(
        "PUBLICATION_LINT_CORRECTION_APPLIED",
        previous_safe_fixes_preserved=True,
        corrected_rules=["UP022", "UP012"],
        recreated_branches=0,
        research_files_written=0,
    )
    return state


def recover_partial_safe_fixes(directory, value, state, head):
    """Validate the user's actual partially fixed worktree before changing it."""
    prior_state_bytes = read(STATE_FILE, MAX_STATE)
    if sha(prior_state_bytes) != PRIOR_STATE_SHA:
        raise Stop("The original failed continuation state differs from the supplied receipt.")
    if head != PIN or state.get("helper_sha256") != PRIOR_PUBLISHER_SHA:
        raise Stop("The failed publication is not at the reviewed source/branch state.")
    old_run = STORE / ("run-" + PRIOR_RUN_ID)
    old_report = read(old_run / "result.json", MAX_STATE)
    if sha(old_report) != PRIOR_REPORT_SHA:
        raise Stop("The original failure receipt differs; no automatic migration.")
    failed = json.loads(old_report)
    commands = failed.get("commands", [])
    if (
        not commands
        or commands[-1].get("label") != "safe_lint_corrections"
        or commands[-1].get("exit_code") != 1
    ):
        raise Stop("This recovery applies only to the documented UP022 lint stop.")
    old_log = read(old_run / "015_safe_lint_corrections.log", MAX_LOG)
    if sha(old_log) != PRIOR_LINT_SHA:
        raise Stop("The saved lint output is not the supplied one.")
    if state.get("steps") != ["RECONCILED_EXISTING_WORKTREE"] or state.get("status") != "PREPARING":
        raise Stop("Unexpected original preparation phase; preserve its state.")
    if git("diff", "--cached", "--name-only", "-z", label="recovery_existing_index").strip("\0\n"):
        raise Stop("Staged changes appeared after the supplied failure; no index is reset.")
    check_files(state["protected_files"])
    backup = STORE / "preserved_review_files.zip"
    if sha(read(backup, 250 * 1024**2)) != state.get("backup_sha256"):
        raise Stop("The original verified publication backup changed or is missing.")
    with zipfile.ZipFile(backup) as archive:
        for name, expected in state["before_files"].items():
            if sha(archive.read(name)) != expected:
                raise Stop("Original backup member changed: " + name)
        old_basis = visual_basis(archive.read("reports/manual_research/index.json"))
    recovery = STORE / ("run-" + RUN_REPORT["run_id"]) / "lint_recovery"
    recovery.mkdir(parents=True, exist_ok=False)
    atomic(recovery / "previous_state.json", prior_state_bytes)
    atomic(recovery / "previous_result.json", old_report)
    atomic(recovery / "previous_lint.log", old_log)
    atomic(recovery / "commodity_publish_previous.py.txt", original_publisher_bytes())
    scratch = recovery / "candidate"
    scratch.mkdir()
    approved = {entry["path"] for entry in value["entries"]}
    python_paths = sorted([*SPEC["patches"], "scripts/commodity_publish.py"])
    archives = {ARCHIVE_ROOT + "/" + name + ".txt" for name in SPEC["patches"]}
    managed = approved | archives | {"scripts/commodity_publish.py"}
    current_changes = status()
    if set(current_changes) - managed:
        raise Stop("Unexpected review files prevent recovery; nothing was overwritten.")
    records = []
    # Recreate the exact pre-safe-fix inputs in scratch, including the original
    # publisher. The old publisher is never executed. All config/source inputs
    # are copied as data; no package or research model is imported here.
    for name in state["protected_files"]:
        if name.startswith(("src/", "tests/", "scripts/")) and name.endswith(".py"):
            atomic(safe(scratch, name), read(safe(REVIEW, name)))
        elif name in {"pyproject.toml", ".python-version", "ruff.toml", ".ruff.toml"}:
            atomic(safe(scratch, name), read(safe(REVIEW, name)))
    for entry in value["entries"]:
        name = entry["path"]
        if name in SPEC["patches"]:
            archive_name = ARCHIVE_ROOT + "/" + name + ".txt"
            original = read(safe(REVIEW, archive_name))
            if sha(original) != SPEC["patches"][name]["input_sha256"]:
                raise Stop("Archived original execution bytes differ: " + name)
            atomic(safe(scratch, name), patched(original, SPEC["patches"][name]))
            reasons = [edit["reason"] for edit in SPEC["patches"][name]["edits"]]
            if name == "scripts/commodity_public_update.py":
                reasons.append(
                    "Replace the reviewed stdout/stderr PIPE pair with capture_output=True (UP022)."
                )
            records.append(
                {
                    "path": name,
                    "original_sha256": sha(original),
                    "archive_path": archive_name,
                    "targeted_changes": reasons,
                }
            )
        else:
            expected = (
                state["before_files"][name]
                if SPEC["classifications"][name] == "NOTEBOOK_OUTPUT_OR_METADATA_CHANGE_ONLY"
                else entry["published_sha256"]
            )
            if sha(read(safe(REVIEW, name))) != expected:
                raise Stop("A preserved document/notebook changed since reconciliation: " + name)
    atomic(scratch / "scripts/commodity_publish.py", original_publisher_bytes())
    run([PYTHON, "-m", "ruff", "--version"], "installed_ruff_version", 15, cwd=REVIEW)
    rows, exit_code = lint_json(
        python_paths, "reconstruct_prior_safe_fixes", scratch, fix=True, selected=FORMAT_SELECT
    )
    expected_partial_finding(rows, exit_code, scratch)
    mismatches = [
        name
        for name in python_paths
        if sha(read(safe(scratch, name))) != sha(read(safe(REVIEW, name)))
    ]
    if mismatches:
        differences = []
        for name in mismatches:
            expected_body = read(safe(scratch, name))
            actual_body = read(safe(REVIEW, name))
            diff = list(
                difflib.unified_diff(
                    expected_body.decode("utf-8").splitlines(),
                    actual_body.decode("utf-8").splitlines(),
                    fromfile="reconstructed/" + name,
                    tofile="review/" + name,
                    lineterm="",
                )
            )
            differences.append(
                {
                    "path": name,
                    "expected_sha256": sha(expected_body),
                    "actual_sha256": sha(actual_body),
                    "diff_truncated": len(diff) > 400,
                    "diff": redact("\n".join(diff[:400])),
                }
            )
        atomic(recovery / "mismatches.json", encode(differences))
        raise Stop(
            "Current review source differs from reconstructed partial fixes: "
            + ", ".join(mismatches)
        )
    event(
        "PARTIAL_SAFE_FIXES_MATCHED",
        python_files=len(python_paths),
        prior_remaining_findings=1,
        retained_notebook_edits=3,
    )
    before = file_map(managed)
    snapshot = recovery / "before_up022.zip"
    with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(managed):
            archive.writestr(name, read(safe(REVIEW, name)))
    with zipfile.ZipFile(snapshot) as archive:
        for name, expected in before.items():
            if sha(archive.read(name)) != expected:
                raise Stop("Recovery snapshot member read-back failed: " + name)
    target = scratch / "scripts/commodity_public_update.py"
    atomic(target, capture_output_correction(read(target)))
    # Publish the corrected continuation, not the previously failed implementation.
    atomic(scratch / "scripts/commodity_publish.py", read(Path(__file__)))
    # The complete publisher, including PR-body construction and its tests,
    # participates in the full configured gate before installation.
    normalize_publication_candidate(python_paths, scratch)
    planned = file_map(python_paths, root=scratch)
    state.update(
        helper_sha256=sha(read(Path(__file__))),
        recovery_version=RECOVERY_VERSION,
        managed_paths=sorted(managed),
        archive_records=records,
        portfolio_inputs_changed=old_basis
        != visual_basis(read(directory / "overlay/reports/manual_research/index.json")),
        portfolio_refreshed=False,
    )
    state["repair"] = {
        "phase": "EXACT_PATCH_PLANNED",
        "created_utc": now(),
        "previous_helper_sha256": PRIOR_PUBLISHER_SHA,
        "original_failure_run": PRIOR_RUN_ID,
        "previous_state_sha256": PRIOR_STATE_SHA,
        "before_files": before,
        "planned_files": planned,
        "candidate_relative": str(scratch.relative_to(STORE)),
        "snapshot_sha256": sha(read(snapshot, 250 * 1024**2)),
        "snapshot_path": str(snapshot),
        "original_backup_preserved": True,
        "partial_safe_fixes_reproduced_exactly": True,
        "candidate_lint_gate": RUN_REPORT.get("candidate_gate"),
    }
    save_state(state)
    return publish_transaction_patch(state)


def quality_checkpoint(state, command, label, seconds, paths, environment):
    """Reuse only a successful exact-input check; never convert failures to passes."""
    inputs = sha(encode(file_map(paths)))
    previous = state.setdefault("quality_checkpoints", {}).get(label)
    argv = [str(value) for value in command]
    if previous and previous.get("input_sha256") == inputs and previous.get("argv") == argv:
        if previous.get("exit_code") != 0:
            raise Stop("A failed quality checkpoint cannot be reused as a pass.")
        if sha(read(Path(previous["log"]), MAX_LOG)) != previous.get("log_sha256"):
            raise Stop("A saved successful quality log changed.")
        event("QUALITY_CHECK_REUSED", check=label)
        return
    run(command, label, seconds, env=environment)
    if sha(encode(file_map(paths))) != inputs:
        raise Stop("Quality command altered a publication input: " + label)
    entry = RUN_REPORT["commands"][-1]
    state["quality_checkpoints"][label] = {
        "input_sha256": inputs,
        "argv": argv,
        "exit_code": 0,
        "log": entry["log"],
        "log_sha256": entry["log_sha256"],
        "passed_utc": now(),
    }
    save_state(state)


def complete_recovered_preparation(directory, value, state):
    approved = set(state["managed_paths"])
    records = state["archive_records"]
    for record in records:
        record["published_sha256"] = sha(read(REVIEW / record["path"]))
    mapping = {
        "publication": "manual-research-20260914",
        "original_source_pin": PIN,
        "source_manifest_sha256": SPEC["manifest_sha256"],
        "files": records,
        "model_parameters_changed": False,
        "historical_experiment_reexecuted": False,
        "scope": "Publication corrections only; original execution-source archives are unchanged.",
    }
    atomic(REVIEW / PUBLIC_PROVENANCE, encode(mapping))
    atomic(REVIEW / PUBLIC_GUIDE, PROVENANCE_TEXT.encode())
    approved.update({PUBLIC_PROVENANCE, PUBLIC_GUIDE})
    state["managed_paths"] = sorted(approved)
    if state["portfolio_inputs_changed"] and not state.get("portfolio_refreshed"):
        overview_refresh(read(directory / "overlay/notebooks/portfolio_overview.ipynb"), state)
    else:
        event("EXECUTED_PORTFOLIO_PRESERVED", charts=6, aggregate_inputs_unchanged=True)
    state["repair"]["phase"] = "QUALITY_PENDING"
    state["repair"]["quality_input_files"] = file_map(approved)
    save_state(state)
    environment = {
        "PYTHONPATH": str(REVIEW / "scripts") + os.pathsep + str(REVIEW / "src"),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    all_paths = set(state["protected_files"]) | approved
    # All repository gates are unchanged. This is intentionally strict.
    quality_checkpoint(
        state,
        [PYTHON, "-u", "scripts/quality.py"],
        "repository_quality",
        660,
        all_paths,
        environment,
    )
    workflow = read(REVIEW / ".github/workflows/quality.yml").decode()
    verifiers = sorted(set(re.findall(r"python\s+(scripts/verify_[A-Za-z0-9_]+\.py)", workflow)))
    if not verifiers:
        raise Stop("Existing workflow verifier commands could not be identified.")
    for verifier in verifiers:
        safe(REVIEW, verifier)
        quality_checkpoint(
            state, [PYTHON, "-u", verifier], Path(verifier).stem, 90, all_paths, environment
        )
    for name, count in [
        ("notebooks/18_released_sequence_ablation.ipynb", 10),
        ("notebooks/19_prior_error_memory_ablation.ipynb", 10),
        ("notebooks/portfolio_overview.ipynb", 6),
    ]:
        notebook_gate(read(REVIEW / name), read(directory / "overlay" / name), count)
    source_gate(value)
    check_files(state["protected_files"])
    if set(status()) - approved:
        raise Stop("Unexpected review output appeared; no files are deleted or staged.")
    for name in approved:
        scan(read(safe(REVIEW, name)), name)
    state.update(
        status="READY_TO_COMMIT",
        final_files=file_map(approved),
        quality_passed=True,
        publication_verifiers=verifiers,
        finished_utc=now(),
    )
    state["repair"]["phase"] = "COMPLETE"
    save_state(state)
    atomic(STORE / "approved_paths.nul", b"".join(n.encode() + b"\0" for n in sorted(approved)))
    review = (
        "Public repository: "
        + REPO
        + "\nExisting branch: "
        + BRANCH
        + "\nNo raw data, environments or trained weights are included.\n"
        + "Review source, notebook outputs and original-byte archives before --publish.\n\n"
        + "\n".join(sorted(approved))
        + "\n"
    )
    atomic(STORE / "review.txt", review.encode())
    atomic(STORE / "PULL_REQUEST.md", PULL_REQUEST_BODY)
    event("READY_TO_COMMIT", files=len(approved), review=str(STORE / "review.txt"))
    return state


def prepare():
    if not STATE_FILE.exists():
        raise Stop(
            "This revision expects the saved failed publication transaction; do not recreate it."
        )
    directory, value = manifest()
    source_gate(value)
    head = assert_repo(REVIEW, BRANCH)
    state = jread(STATE_FILE)
    helper_hash = sha(read(Path(__file__)))
    if state.get("helper_sha256") == helper_hash:
        if state.get("status") in {
            "READY_TO_COMMIT",
            "COMMITTED",
            "PUSHED",
            "PR_OPEN",
            "MERGED",
            "MAIN_PUBLICATION_VERIFIED",
        }:
            check_files(state["final_files"])
            check_files(state["protected_files"])
            event("PREPARATION_REUSED", new_tests=0, new_fits=0)
            return state
        if state.get("recovery_version") != RECOVERY_VERSION:
            raise Stop("Unknown partial recovery transaction.")
        phase = state.get("repair", {}).get("phase")
        if phase == "EXACT_PATCH_PLANNED":
            state = publish_transaction_patch(state)
        elif phase == "EXACT_PATCH_APPLIED":
            check_files(state["repair"]["installed_files"])
        else:
            raise Stop(
                "A later preparation gate stopped. Return its ZIP; do not repeat an unknown failure."
            )
    else:
        if state.get("helper_sha256") != PRIOR_PUBLISHER_SHA:
            raise Stop("Unexpected publisher version; preserve its saved state.")
        run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--self-test"],
            "lint_recovery_regression_tests",
            60,
            cwd=Path.home(),
        )
        if not PYTHON.is_file():
            raise Stop(
                "The existing verified interpreter is unavailable; do not install a replacement."
            )
        state = recover_partial_safe_fixes(directory, value, state, head)
    return complete_recovered_preparation(directory, value, state)


def ready_state():
    state = jread(STATE_FILE)
    if state.get("helper_sha256") != sha(read(Path(__file__))):
        raise Stop(
            "Publication helper changed after its quality transaction; do not reuse its gate."
        )
    if not state.get("quality_passed") or state.get("status") not in {
        "READY_TO_COMMIT",
        "COMMITTED",
        "PUSHED",
        "PR_OPEN",
        "MERGED",
        "MAIN_PUBLICATION_VERIFIED",
    }:
        raise Stop("The continuation quality gate has not passed; do not stage or push.")
    check_files(state["final_files"])
    check_files(state["protected_files"])
    _, value = manifest()
    source_gate(value)
    found = status()
    if set(found) - set(state["final_files"]):
        raise Stop("Unexpected new worktree changes require review before publication.")
    for name in state["final_files"]:
        scan(read(safe(REVIEW, name)), name)
    return state


def indexed_gate(state):
    expected = state["final_files"]
    changed = {
        x
        for x in git("diff", "--cached", "--name-only", "-z", label="staged_paths").split("\0")
        if x
    }
    if not changed or changed - set(expected):
        raise Stop("Empty or unexpected staged file list.")
    if git("diff", "--cached", "--diff-filter=DR", "--name-only", label="staged_deletions").strip():
        raise Stop("Unexpected staged deletion/rename.")
    entries = git("ls-files", "--stage", "-z", label="staged_objects").split("\0")
    indexed = {}
    for entry in entries:
        if entry:
            metadata, path = entry.split("\t", 1)
            mode, digest, stage = metadata.split()
            indexed[path] = (mode, digest, stage)
    for path in expected:
        mode, digest, stage = indexed[path]
        body = read(REVIEW / path)
        blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if mode not in {"100644", "100755"} or stage != "0" or digest != blob:
            raise Stop("Index and reviewed bytes differ: " + path)
    git("diff", "--cached", "--check", label="staged_whitespace_check")
    git("diff", "--cached", "--stat", label="staged_summary")


def gh_available():
    return shutil.which("gh") is not None


def open_pr(state):
    if not gh_available():
        state["status"] = "PUSHED"
        state["compare_url"] = COMPARE_URL
        save_state(state)
        event("PUSHED_OPEN_PR_IN_BROWSER", compare_url=COMPARE_URL, commit=state["commit"])
        return state
    try:
        text = run(
            [
                "gh",
                "pr",
                "list",
                "-R",
                REPO,
                "--state",
                "open",
                "--base",
                "main",
                "--head",
                BRANCH,
                "--json",
                "number,url,headRefOid",
            ],
            "find_existing_pr",
            30,
        )
        prs = json.loads(text)
        if len(prs) > 1:
            raise Stop("Multiple matching pull requests require review.")
        if prs:
            pr = prs[0]
            if pr["headRefOid"] != state["commit"]:
                raise Stop("Existing PR head does not match the pushed commit.")
            state["pull_request"] = pr
        else:
            url = run(
                [
                    "gh",
                    "pr",
                    "create",
                    "-R",
                    REPO,
                    "--base",
                    "main",
                    "--head",
                    BRANCH,
                    "--title",
                    TITLE,
                    "--body-file",
                    str(STORE / "PULL_REQUEST.md"),
                ],
                "create_pull_request",
                60,
            ).strip()
            state["pull_request"] = {"url": url}
        state["status"] = "PR_OPEN"
        save_state(state)
        event("PUSHED_PR_OPEN", commit=state["commit"], pull_request=state["pull_request"]["url"])
    except (Stop, ValueError, KeyError) as exc:
        # A successful push is never repeated just to recover PR creation.
        state["status"] = "PUSHED"
        state["pr_error"] = redact(str(exc))
        state["compare_url"] = COMPARE_URL
        save_state(state)
        event(
            "PUSHED_OPEN_PR_IN_BROWSER",
            compare_url=COMPARE_URL,
            commit=state["commit"],
            message="Branch is pushed. Use the browser to open or find its pull request.",
        )
    return state


def publish(ack):
    if not ack:
        raise Stop(
            "Review the listed public source/notebook outputs, then pass --reviewed-public-code."
        )
    state = ready_state()
    head = assert_repo(REVIEW, BRANCH)
    if state.get("commit"):
        if head != state["commit"]:
            raise Stop("Review HEAD changed after the recorded commit.")
    else:
        if head != state["base_head"]:
            # Recover only our exact committed tree and message after a journal interruption.
            message = git(
                "show", "-s", "--format=%s", "HEAD", label="recovery_commit_subject"
            ).strip()
            parent = git("rev-parse", "HEAD^", label="recovery_commit_parent").strip()
            recovered_tree = git("rev-parse", "HEAD^{tree}", label="recovery_commit_tree").strip()
            if (
                message != TITLE
                or parent != state["base_head"]
                or recovered_tree != state.get("expected_tree")
                or status()
            ):
                raise Stop("Unknown commit appeared on the review branch; preserve it.")
            state["commit"] = head
            state["status"] = "COMMITTED"
            save_state(state)
        else:
            # Check identity before staging. No guessed author or email.
            git("var", "GIT_AUTHOR_IDENT", label="commit_identity")
            git(
                "fetch",
                "--no-tags",
                "origin",
                "refs/heads/main:refs/remotes/origin/main",
                label="fetch_main_before_commit",
                seconds=60,
            )
            main = git("rev-parse", "origin/main", label="fetched_main").strip()
            if main != state["base_head"]:
                raise Stop(
                    "Remote main advanced after the tested base. Do not force-push or merge untested changes."
                )
            staged = {
                x
                for x in git(
                    "diff", "--cached", "--name-only", "-z", label="existing_index_changes"
                ).split("\0")
                if x
            }
            if staged - set(state["final_files"]):
                raise Stop("Unexpected files are staged; no index reset will be performed.")
            git(
                "add",
                "-f",
                "--pathspec-from-file=" + str(STORE / "approved_paths.nul"),
                "--pathspec-file-nul",
                label="stage_approved_paths",
                seconds=60,
            )
            indexed_gate(state)
            state["expected_tree"] = git("write-tree", label="expected_commit_tree").strip()
            save_state(state)
            git("commit", "-m", TITLE, label="commit_reviewed_publication", seconds=60)
            commit = git("rev-parse", "HEAD", label="committed_head").strip()
            tree = git("rev-parse", "HEAD^{tree}", label="committed_tree").strip()
            if tree != state["expected_tree"]:
                raise Stop("A commit hook changed the reviewed tree; do not push it.")
            state.update(commit=commit, status="COMMITTED", committed_utc=now())
            save_state(state)
    if status():
        raise Stop("Review worktree changed after commit. Commit is preserved; push is blocked.")
    if not state.get("push_verified"):
        git("push", "--set-upstream", "origin", BRANCH, label="push_publication_branch", seconds=90)
        refs = (
            git(
                "ls-remote",
                "origin",
                "refs/heads/" + BRANCH,
                label="verify_remote_branch",
                seconds=30,
            )
            .strip()
            .split()
        )
        if len(refs) != 2 or refs[0] != state["commit"]:
            raise Stop("Could not confirm the pushed branch SHA.")
        state.update(status="PUSHED", push_verified=True, pushed_utc=now())
        save_state(state)
    return open_pr(state)


def pr_view(state):
    target = state.get("pull_request", {}).get("url") or BRANCH
    value = run(
        [
            "gh",
            "pr",
            "view",
            target,
            "-R",
            REPO,
            "--json",
            "url,number,state,isDraft,headRefOid,baseRefName,mergeStateStatus,reviewDecision,statusCheckRollup,mergeCommit",
        ],
        "pull_request_state",
        30,
    )
    return json.loads(value)


def merge():
    state = ready_state()
    if not state.get("push_verified"):
        raise Stop("No verified pushed publication commit.")
    if not gh_available():
        event(
            "MERGE_IN_BROWSER",
            compare_url=COMPARE_URL,
            message="Open the PR, require green checks on this commit, and use Create a merge commit. Then run --verify-merge.",
        )
        return state
    pr = pr_view(state)
    if pr.get("headRefOid") != state["commit"] or pr.get("baseRefName") != "main":
        raise Stop("PR head/base differs from the verified publication.")
    if pr.get("state") == "MERGED":
        state["status"] = "MERGED"
        state["pull_request"] = {"url": pr["url"], "number": pr["number"]}
        save_state(state)
        return verify_merge()
    if pr.get("isDraft") or pr.get("state") != "OPEN":
        raise Stop("PR is not an open, ready-for-review publication.")
    checks = pr.get("statusCheckRollup") or []
    successful_quality = False
    for check in checks:
        name = check.get("name", check.get("context", ""))
        conclusion = check.get("conclusion") or check.get("state")
        if conclusion not in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
            raise Stop(
                "A PR check is pending or failed: "
                + name
                + ". Do not merge; report captures its state."
            )
        if "quality" in name.lower() and conclusion == "SUCCESS":
            successful_quality = True
    if not successful_quality:
        raise Stop("A successful Quality check on this exact PR head has not been established.")
    if (
        pr.get("mergeStateStatus") not in {"CLEAN", "HAS_HOOKS"}
        or pr.get("reviewDecision") == "CHANGES_REQUESTED"
    ):
        raise Stop(
            "GitHub has unresolved merge/review requirements. No administrator bypass is used."
        )
    run(
        [
            "gh",
            "pr",
            "merge",
            str(pr["number"]),
            "-R",
            REPO,
            "--merge",
            "--match-head-commit",
            state["commit"],
        ],
        "normal_guarded_pr_merge",
        60,
    )
    observed = pr_view(state)
    if observed.get("state") != "MERGED":
        raise Stop("GitHub has not confirmed the PR as merged. No merge completion is claimed.")
    state.update(
        status="MERGED",
        pull_request={"url": observed["url"], "number": observed["number"]},
        merge_commit=(observed.get("mergeCommit") or {}).get("oid"),
    )
    save_state(state)
    return verify_merge()


def verify_merge():
    state = ready_state()
    if not state.get("commit"):
        raise Stop("No recorded publication commit to verify.")
    if status():
        raise Stop(
            "Review worktree has newer edits; preserve them before fast-forward verification."
        )
    git(
        "fetch",
        "--no-tags",
        "origin",
        "refs/heads/main:refs/remotes/origin/main",
        label="fetch_main_after_merge",
        seconds=60,
    )
    main = git("rev-parse", "origin/main", label="merged_main_head").strip()
    git("merge-base", "--is-ancestor", state["commit"], main, label="publication_commit_in_main")
    rows = git("ls-tree", "-r", "-z", main, label="published_main_tree").split("\0")
    tree = {}
    for row in rows:
        if row:
            metadata, name = row.split("\t", 1)
            mode, kind, obj = metadata.split()
            tree[name] = (kind, obj, mode)
    for name in state["final_files"]:
        body = read(REVIEW / name)
        expected = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if name not in tree or tree[name][0] != "blob" or tree[name][1] != expected:
            raise Stop("Main does not contain the exact reviewed bytes: " + name)
    git("merge", "--ff-only", "origin/main", label="synchronize_review_worktree", seconds=30)
    if status():
        raise Stop("Unexpected review changes after fast-forward; preserved.")
    if gh_available():
        try:
            observed = pr_view(state)
            state["github_pr_state"] = observed.get("state")
        except (Stop, ValueError):
            state["github_pr_state"] = "UNVERIFIED"
    state.update(
        status="MAIN_PUBLICATION_VERIFIED",
        main_commit=main,
        verified_utc=now(),
        published_files_verified=len(state["final_files"]),
        research_source_unchanged=True,
        review_worktree_matches_main=True,
        scope="Exact publication bytes in fetched main; active research checkout retains its execution pin. Post-merge CI is not inferred.",
    )
    save_state(state)
    event("MAIN_PUBLICATION_VERIFIED", main_commit=main, files=state["published_files_verified"])
    return state


def export_report():
    """Bounded return archive; excludes original source backups and datasets."""
    export_end = time.monotonic() + 15
    atomic(REPORT, encode(RUN_REPORT))
    temporary = BUNDLE.with_suffix(".zip.part")
    size = 0
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", read(REPORT, MAX_STATE))
        if STATE_FILE.exists():
            state = jread(STATE_FILE)
            # Hash maps, stage receipts and commits; never copy source, data or credentials.
            archive.writestr("state.json", encode(state))
        history_logs = sorted(
            STORE.glob("run-*/*.log"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for index, path in enumerate(history_logs):
            if time.monotonic() > export_end or size > 48 * 1024**2:
                archive.writestr(
                    "EXPORT_LIMIT.txt", "Remaining logs stay in the printed local run directory.\n"
                )
                break
            if path.exists() and not path.is_symlink() and path.stat().st_size <= MAX_LOG:
                body = redact(read(path, MAX_LOG).decode("utf-8", errors="replace")).encode()
                archive.writestr(f"logs/{index:03d}_{path.name}", body)
                size += len(body)
        for path in sorted(STORE.glob("run-*/result.json")):
            if time.monotonic() > export_end:
                break
            archive.writestr("receipts/" + path.parent.name + ".json", read(path, MAX_STATE))
        diagnostic_files = sorted(STORE.glob("run-*/*.json"))
        diagnostic_files += sorted(STORE.glob("run-*/lint_recovery/*.json"))
        for index, path in enumerate(diagnostic_files):
            if time.monotonic() > export_end or size > 48 * 1024**2:
                break
            if path.name == "result.json" or path.is_symlink() or path.stat().st_size > MAX_STATE:
                continue
            body = redact(read(path, MAX_STATE).decode("utf-8", errors="replace")).encode()
            archive.writestr(f"diagnostics/{index:03d}_{path.name}", body)
            size += len(body)
        if RUN_REPORT.get("status") == "STOPPED":
            source_receipts = []
            for name in sorted([*SPEC["patches"], "scripts/commodity_publish.py"]):
                if time.monotonic() > export_end or size > 48 * 1024**2:
                    break
                path = safe(REVIEW, name)
                if not path.is_file() or path.stat().st_size > 2 * 1024**2:
                    continue
                original = read(path, 2 * 1024**2)
                body = redact(original.decode("utf-8", errors="replace")).encode()
                archive.writestr("failure_sources/" + name + ".txt", body)
                source_receipts.append(
                    {
                        "path": name,
                        "original_sha256": sha(original),
                        "included_sha256": sha(body),
                        "redacted": body != original,
                    }
                )
                size += len(body)
            archive.writestr("failure_sources/index.json", encode(source_receipts))

        candidate = RUN_REPORT.get("candidate_snapshot", {})
        if RUN_REPORT.get("status") == "STOPPED" and candidate:
            candidate_root = Path(candidate["root"])
            candidate_index = []
            for name in candidate.get("python_paths", []):
                if time.monotonic() > export_end or size > 48 * 1024**2:
                    archive.writestr(
                        "CANDIDATE_EXPORT_LIMIT.txt",
                        "Some candidate sources remain in the recorded local scratch directory.\n",
                    )
                    break
                path = safe(candidate_root, name)
                if not path.is_file() or path.stat().st_size > 2 * 1024**2:
                    continue
                raw = read(path, 2 * 1024**2)
                body = redact(raw.decode("utf-8", errors="replace")).encode()
                archive.writestr("candidate_sources/" + name + ".txt", body)
                candidate_index.append(
                    {
                        "path": name,
                        "original_sha256": sha(raw),
                        "included_sha256": sha(body),
                        "redacted": body != raw,
                    }
                )
                size += len(body)
            archive.writestr("candidate_sources/index.json", encode(candidate_index))

        tests = STORE / "self_test.json"
        if tests.exists():
            archive.writestr("continuation_tests.json", read(tests, MAX_STATE))
    os.replace(temporary, BUNDLE)
    print("RETURN_ZIP:", BUNDLE, flush=True)
    print("REPORT:", REPORT, flush=True)


def main():
    global DEADLINE, RUN_REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--publish", action="store_true")
    group.add_argument("--merge", action="store_true")
    group.add_argument("--verify-merge", action="store_true")
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--reviewed-public-code", action="store_true")
    args = parser.parse_args()
    mode = next(
        k for k in ("prepare", "publish", "merge", "verify_merge", "self_test") if getattr(args, k)
    )
    limit = {"prepare": 1140, "publish": 300, "merge": 180, "verify_merge": 150, "self_test": 45}[
        mode
    ]
    DEADLINE = time.monotonic() + limit
    STORE.mkdir(parents=True, exist_ok=True)
    RUN_REPORT = {
        "project": REPO,
        "mode": mode,
        "status": "STARTED",
        "started_utc": now(),
        "run_id": datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ"),
        "helper_sha256": sha(read(Path(__file__))),
        "limit_seconds": limit,
        "publisher_revision": RECOVERY_VERSION,
        "research_models_requested": 0,
        "cloud_operations": 0,
        "commands": [],
        "preparation_execution_claim": "User-run; not previously executed by the assistant.",
    }
    atomic(REPORT, encode(RUN_REPORT))
    code = 0
    lock = None
    try:
        if mode != "self_test":
            event("PUBLISHER_REVISION", revision=RECOVERY_VERSION)
            lock = (STORE / "operation.lock").open("a")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if mode == "self_test":
            result = self_test()
        elif mode == "prepare":
            result = prepare()
        elif mode == "publish":
            result = publish(args.reviewed_public_code)
        elif mode == "merge":
            result = merge()
        else:
            result = verify_merge()
        RUN_REPORT["status"] = result.get("status", "COMPLETE")
        RUN_REPORT["result_summary"] = {
            k: result[k]
            for k in (
                "status",
                "commit",
                "main_commit",
                "pull_request",
                "compare_url",
                "published_files_verified",
                "review_worktree_matches_main",
            )
            if k in result
        }
        print("RESULT:", RUN_REPORT["status"], flush=True)
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        code = 1
        RUN_REPORT.update(
            status="STOPPED",
            error=redact(type(exc).__name__ + ": " + str(exc)),
            traceback=redact(traceback.format_exc()),
        )
        print("RESULT: STOPPED\nERROR: " + RUN_REPORT["error"], flush=True)
        print(
            "Do not reset, recreate the branch, delete checkpoints or rerun a failed step unchanged.",
            flush=True,
        )
    finally:
        RUN_REPORT["finished_utc"] = now()
        RUN_REPORT["logs"] = [str(p) for p in LOGS]
        atomic(STORE / ("run-" + RUN_REPORT["run_id"]) / "result.json", encode(RUN_REPORT))
        if lock is not None:
            lock.close()
        try:
            export_report()
        except BaseException as exc:
            print("REPORT_EXPORT_WARNING:", type(exc).__name__, str(exc), flush=True)
            print(
                "Use commodity_publish_report.json and commodity_publish_bootstrap.log instead.",
                flush=True,
            )
    return code


# Evidence payload is inert text, never evaluated as code.
SPEC = json.loads(
    '{\n  "release": "release-20260914T010637288683Z",\n  "manifest_sha256": "25944bd1af23693f771c7eeeb2cb81723f92764ffabf0246eb125b5bd5a839ea",\n  "review_head": "d142a4cb57a5c4b2880f9341619e13a735b1cddc",\n  "branch": "chore/public-research-update-20260913",\n  "review_hashes": {\n    "AGENTS.md": "577f4646b492b7f0ef9f0abb366c1135e4eb1b213350ea9ad8d0be8ba8ff074a",\n    "README.md": "618e0436772e7f1af4d08be901454e2bbd81106d87289fce8a1f9c5968bea4a8",\n    "configs/manual_close_network.json": "70b2fbc653b1abff891b5da01bb2ee22d8b8a4889db42a659c1064d755c35748",\n    "configs/manual_event_history.json": "9d3de4393a9d4fbd848b3afbefcf2050e2fcc68047e09ef92b474f47536ca789",\n    "configs/manual_feature_diagnosis.json": "4398baf1542e8312f352ef7f5b41bdf33f3cf8b48d53fb033b3de9888a4d5d70",\n    "configs/manual_information_audit.json": "c903443f0e6185cb7f18535f22e54680c3ae1cc0fd4f511553a975b1fca8e4bc",\n    "configs/manual_innovation_ablation.json": "d1083c984c6054cd2812fcd16ab47730cda232dc10f3c70baaa038ea676434d8",\n    "configs/manual_prior_dynamics.json": "c4285173e5e068cb9706ffa14d88aedcec92f1d5fb08eb384830a090836d3492",\n    "configs/manual_prior_error_memory.json": "3d7ad16cc757fcfbd08e07bef8e928cd5c50d1895515b03642430ef390deefa0",\n    "configs/manual_prior_replication.json": "cbf601aa791e1cf6734ce649c0e4d26cc906b47f295590c0921b5eb441c8a264",\n    "configs/manual_rank_state_ablation.json": "fc32bc35c33925fc5613b1b3a720c5f73114d4180586c8aac5e7176183c5d19c",\n    "configs/manual_response_encoding.json": "128b16b6da292786a9ade4b81cfd96544be0310841ef18142899cff594f2ad95",\n    "configs/manual_sequence_state.json": "4ac1fa383022463988099d382b37a4614071f565e53bb9e55c28957f06c565a7",\n    "configs/manual_session_ablation.json": "dcdd72bd935900ff62453fda1c59e64c815ce9dd2875c66da9cf64ac47d5f01b",\n    "configs/manual_target_context.json": "c59e97d66e68a970a5db0021c450c2e7ca6ffb3743f2925fc7c10578b59d21d5",\n    "docs/MANUAL_REPRODUCTION.md": "57a1889621ab6f24312b8f344f94f151215ec189748e0922cfc936cb6e9020e7",\n    "docs/MANUAL_RESEARCH_RESULTS.md": "8031280949e603cb9fcfdc0fc8a7c6304a29bd8e7ced0c61d3d709f1a0f33d3f",\n    "docs/PUBLICATION_SCOPE.md": "c11f702f8a06feccfa5360e2335dd5ce5e13cd92110f364b9e40843fe8987fdd",\n    "docs/RECOVERY_AND_RUN_ORDER.md": "42b2607d0e4b8f1b1237eb712976f2cf74ee7ad88ed8f54e6c87d1f966b94c99",\n    "docs/ROUND_16_PROTOCOL.md": "cee4c1800259012a9c206c6079ea3dc68b37a898342b46a694635f593001007d",\n    "docs/ROUND_17_PROTOCOL.md": "6b388bb1525782326fc84b2e4e077aca5c0366160a6bfbd7a875d3931a6da323",\n    "docs/ROUND_18_PROTOCOL.md": "319127afb0fdcd97919356243c525ceed970837b5ae37e73b4c14aafcf682e1d",\n    "docs/ROUND_19_PROTOCOL.md": "84f4149c51248f88818d771d5a146d217d3c63d886328615cd9975a6a884266c",\n    "docs/history/AGENTS_before_manual_publication.md": "0b1c2e3216ca313c684cae2fd7dbf69726c9d0704ee394db3eca372559b3eaa6",\n    "docs/history/README_before_manual_publication.md": "51cdafeb2fddaeb9e9395e7d8b86ce7c5b5ef1de3cbaaf6fb4076f8680b10568",\n    "docs/manual_close_network.md": "309d0d130433584b3783b985a885941df2c793f476e5db476dbd0a8224c60fc9",\n    "docs/manual_event_history.md": "fe5f194353f0f3cdc87f4ddb22b901aa925c69e26835cc7a16a03cb2459e5011",\n    "docs/manual_feature_diagnosis.md": "5e9b0bf6841705be5b390d1dcdda4e3bf5be892d203a853f37345eed63e960f4",\n    "docs/manual_information_audit.md": "e2d836cc85c929088069d287c6ddd7ccb7fdcddb0bd2d59131ec56f4ff61aa8f",\n    "docs/manual_innovation_ablation.md": "ee739eef2ece65a29453bdd7257cfb1dde17ccf89a608fb5c16578ae146fe106",\n    "docs/manual_prior_dynamics.md": "46952ed0cf98fd3c048ddb045a7ff9c5c797140a00c6cbdc623eed2444e7c2f4",\n    "docs/manual_prior_replication.md": "ec8fda7de443632ed0b9e6d70dd9d3cf94d26833994f39d06c187d720553a802",\n    "docs/manual_prior_research_git.md": "e816bb5b72d41cd97dba9aae4a71b1f01cf8f8bd653e56ba9a4fc40394e4fab4",\n    "docs/manual_rank_state_ablation.md": "56180f385e3f5ae5aef6dcf9453dc60bea6d4d0ad33113a5c667d6fd231bc1e4",\n    "docs/manual_response_encoding.md": "5339bec63c62402506e59d59c2efd59c0c80d1f061148f245e9ef8ae2111c906",\n    "docs/manual_session_ablation.md": "d7039c055f2cb245b773a524614c77c610e443994034f9018d1021025616cfbc",\n    "docs/manual_target_context.md": "f9dc1681a2b8e1732a919bb52089f3976a97bd22e812543d3b960a4a3880d115",\n    "notebooks/03_manual_readiness.ipynb": "0990f8a7aaf67d8dee8fb1011fac685f417f302298eed99f419523563398ea84",\n    "notebooks/04_normalization_first_fold.ipynb": "b2fbf1ec75ded3de3e7734fd73923b9cd70b2d8f5e2bfaf8b6c28a65084b1e4e",\n    "notebooks/05_normalization_validation.ipynb": "146fa340cd3a6b0a25c63395d717f74092b61390fd58074dc63a3407bd118251",\n    "notebooks/06_session_feature_lab.ipynb": "4d071e9856f56f251c30bd56b9ccc00f78366b05e62a4c748479a9bf21a949b4",\n    "notebooks/07_session_feature_ablation.ipynb": "070deedc081f83926cccc7dcfb4f4903824a071757b99a8ef09bea602e7743da",\n    "notebooks/08_close_network_ablation.ipynb": "f7e5771fc0d9f2ee0851a6fb0846d5975528540371df18c9d6341fc3472a2be5",\n    "notebooks/09_feature_diagnosis_and_rank_lab.ipynb": "99ba837bcc179b0b4e99b355183ee791f6b1b065679d3945c1356731e7eb8553",\n    "notebooks/10_released_rank_ablation.ipynb": "921e77c663a0c12a23f68b4649e4f06caab2b1d7a42afbde5ebf69c33f108376",\n    "notebooks/11_target_context_ablation.ipynb": "a732d7fab9bfae7c16adcac1e461393ec543c184b524840df2014a80007c5da8",\n    "notebooks/12_delayed_response_ablation.ipynb": "9ba02e2f3509c37dbe91cf88ee1cd46d582536007bf20c5a35a8a7086297e799",\n    "notebooks/13_information_audit_and_relative_feature_lab.ipynb": "221ec784ad1267a65953ce80b02e311b6485c951ba46e16362c0c38b1e18e3ff",\n    "notebooks/14_innovation_feature_ablation.ipynb": "4dbf9c4eb59d134aa84f319a2e9e3725365fa0262eea10aae14d9c4f300c0026",\n    "notebooks/15_prior_dynamics_ablation.ipynb": "bb23ef82fa6972c489527ae9a98c7c665800a7989afde504f6302c79195f486a",\n    "notebooks/16_prior_dynamics_replication.ipynb": "765715350c5144898107d38b2b4e1e5ae25f2d8c2645ecb76a95f4792e9b959e",\n    "notebooks/17_event_history_ablation.ipynb": "7c689aa2ed2eafa7d7cb51d8c2f518d854be8b0a694a76ff4cf40839d75453ce",\n    "notebooks/18_released_sequence_ablation.ipynb": "53e2004acbb7b46bf99b8538901155101ca81fcf9e110f4b33af2d20b9afd470",\n    "notebooks/19_prior_error_memory_ablation.ipynb": "cbd9d72c93983983d9522333b39487b2aac628284364585a93f21b976dfdabb9",\n    "notebooks/portfolio_overview.ipynb": "4dbe773fea5ca35923ce156ce5013970116167e53b5bb6a188642ea3593d63aa",\n    "reports/manual_research/development_reference.svg": "515e6c25776d64e1f18be3f5482af4140877f7b310415d698d510efb818d0bbf",\n    "reports/manual_research/index.json": "c2a2a2655d69c016fd337e1b8357dcc9dfbef886923bc6677f463dcc9dc739d8",\n    "reports/manual_research/study_03.json": "67dfd4f74563f79566deb08f8093379a837059333b157ae47528d5a66b323c1f",\n    "reports/manual_research/study_04.json": "c9f77e201df0771fa1057b9baac25e2a6f6e93096f0c88f2b0a05cf138cf830d",\n    "reports/manual_research/study_05.json": "c4a227384025a243e05c602f69768ca6a9ad07e5c682948311d7970c68804796",\n    "reports/manual_research/study_06.json": "fe83f1f29b3291afffbe076379bfb9998c5c9a127f4c072044ead702d699aeff",\n    "reports/manual_research/study_07.json": "0b76f38c7d8c237874e5c6bb690f4961150a64c693999b522ed57b2e03aac992",\n    "reports/manual_research/study_08.json": "4293802bbebca0268d8577ae699fa659c20857a57b31de194edf7a38825c2347",\n    "reports/manual_research/study_09.json": "9eb3b7c2d6b520bde8df13163effeb5995d59a9012d21becd261cd7eca4d647e",\n    "reports/manual_research/study_10.json": "a95c4d66c73f8b10e175f3c0c96545223f0767e263668d5362cc649e10a7691a",\n    "reports/manual_research/study_11.json": "f47f55d23b369db16125bd3b0e401bb8e3dc2619985555918c932f21d4941962",\n    "reports/manual_research/study_12.json": "2e280d6a35bc763887373c861e1ea841ad513cd9dae19d95fa08683399a5429d",\n    "reports/manual_research/study_13.json": "1145282ae9fef6d4fce85b8190bc650c59ef72d2ee36891aefaaac2587a6a6b9",\n    "reports/manual_research/study_14.json": "2a81121963f2ee1592d11f04d445c51dbc8d479e6b8b16a1c63e2be78821d58f",\n    "reports/manual_research/study_15.json": "8ef71a33d373448a1cf54722ecdba3166b69afb5ea0a0b4260a5ca77d708d65e",\n    "reports/manual_research/study_16.json": "70648b06f09670f70049aa66822387536c50bebbad5538fa8d3595286a6dea36",\n    "reports/manual_research/study_17.json": "f5a83613a32a7c761a4d41d6b6837eebaad48d6fddbcb8da54519ba5f3412f87",\n    "reports/manual_research/study_18.json": "88da404c2e9235772d95c495393ee418d98089685ffde4073be01ef9c3c3cc71",\n    "reports/manual_research/study_19.json": "52b5a95bf927df11fec11424e378e120c2dfd60732d0bb43687b8d4811ffa5d8",\n    "scripts/commodity_close_network.py": "3b0c9b77b5e44962bd9c0068c7f958a7418f826943f5c025aeff2261d4cdf8d4",\n    "scripts/commodity_feature_diagnosis.py": "a8991b0e9b5517cf5c6c19659084c999aec3cfb7addb28252470b738df5ef80b",\n    "scripts/commodity_feature_round.py": "fc27479ee4821dec7e4f965e1d32c885f7cad0b157b6e1bb3b1cd7ac9822ea73",\n    "scripts/commodity_first_fold.py": "517c76064e2c70469843b007c446caf7cc8678440d61c095f42feb6faf62b53d",\n    "scripts/commodity_information_audit.py": "e86f608be525d41c667d261cc0c9fa04a51f08e17b2a58d0a81224a9771d27df",\n    "scripts/commodity_next_research.py": "d923cfdbe726d0c068612f5c5066beec4fe631b9919e1425e08de114bcf4dc9d",\n    "scripts/commodity_prior_research.py": "eb3e968efebf49c036bd42f1610efc33444a7706059030cb18de5773285d0cc8",\n    "scripts/commodity_public_update.py": "3ee316a503f19a8b9e04608f4c8042f6b32840e264beed8bb9681e02948bca7a",\n    "scripts/commodity_publication.py": "41f41e98fc81de6e0897970537b3751fc40a5dfe32408ccf8661845cdaafe21f",\n    "scripts/commodity_rank_state_ablation.py": "bce98b5326fad9fd1e1f04d2536629b1641d8428443e09de4ac641ea87a40762",\n    "scripts/commodity_research_release.py": "cc8d68e69aa42fedd95a2002f238d1ff9437ce80dbbed2f4b8ca71e1d08f7827",\n    "scripts/commodity_response_encoding.py": "51db0fcdc93dc75e671c269ade763797f5c555a10ea7d5b0c22d9fd270f7b496",\n    "scripts/commodity_session_ablation.py": "7efef4574d55bd9aad8da8ecf4e7f4ba7cc54907412dcfb4a754100d9e34f88f",\n    "scripts/commodity_target_context.py": "4c9f2c11f76fe6f9f31516ae078610879fb50bad4b4a55ae0ce3a9ca5bfbd2c3",\n    "scripts/commodity_two_feature_rounds.py": "a4284a2a80e42568899704e694b98ef351181be7aa33b3af86135a857a0852aa",\n    "tests/test_manual_next_research.py": "a84438ce5508c56de2106291de70184835e6f83032ed26b5a1124cbd098e8430",\n    "tests/test_manual_prior_research.py": "d5d0475ebac01f3104ac2bbf34dd75fadc47ebced0c086aab545ea9fccabccad",\n    "tests/test_manual_publication.py": "83f3fbceef4214e8788107655cb826e55bb77199c2605645d1f4ab9c85f1e53b"\n  },\n  "classifications": {\n    "AGENTS.md": "MATCHES_CURRENT_PUBLICATION",\n    "README.md": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_close_network.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_event_history.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_feature_diagnosis.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_information_audit.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_innovation_ablation.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_prior_dynamics.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_prior_error_memory.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_prior_replication.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_rank_state_ablation.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_response_encoding.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_sequence_state.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_session_ablation.json": "MATCHES_CURRENT_PUBLICATION",\n    "configs/manual_target_context.json": "MATCHES_CURRENT_PUBLICATION",\n    "docs/MANUAL_REPRODUCTION.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/MANUAL_RESEARCH_RESULTS.md": "MATCHES_EARLIER_PUBLICATION",\n    "docs/PUBLICATION_SCOPE.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/RECOVERY_AND_RUN_ORDER.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/ROUND_16_PROTOCOL.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/ROUND_17_PROTOCOL.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/ROUND_18_PROTOCOL.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/ROUND_19_PROTOCOL.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/history/AGENTS_before_manual_publication.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/history/README_before_manual_publication.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_close_network.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_event_history.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_feature_diagnosis.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_information_audit.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_innovation_ablation.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_prior_dynamics.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_prior_replication.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_prior_research_git.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_rank_state_ablation.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_response_encoding.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_session_ablation.md": "MATCHES_CURRENT_PUBLICATION",\n    "docs/manual_target_context.md": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/03_manual_readiness.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/04_normalization_first_fold.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/05_normalization_validation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/06_session_feature_lab.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/07_session_feature_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/08_close_network_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/09_feature_diagnosis_and_rank_lab.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/10_released_rank_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/11_target_context_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/12_delayed_response_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/13_information_audit_and_relative_feature_lab.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/14_innovation_feature_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/15_prior_dynamics_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/16_prior_dynamics_replication.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/17_event_history_ablation.ipynb": "MATCHES_CURRENT_PUBLICATION",\n    "notebooks/18_released_sequence_ablation.ipynb": "NOTEBOOK_OUTPUT_OR_METADATA_CHANGE_ONLY",\n    "notebooks/19_prior_error_memory_ablation.ipynb": "NOTEBOOK_OUTPUT_OR_METADATA_CHANGE_ONLY",\n    "notebooks/portfolio_overview.ipynb": "NOTEBOOK_OUTPUT_OR_METADATA_CHANGE_ONLY",\n    "reports/manual_research/development_reference.svg": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/index.json": "MATCHES_EARLIER_PUBLICATION",\n    "reports/manual_research/study_03.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_04.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_05.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_06.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_07.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_08.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_09.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_10.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_11.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_12.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_13.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_14.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_15.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_16.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_17.json": "MATCHES_CURRENT_PUBLICATION",\n    "reports/manual_research/study_18.json": "MATCHES_EARLIER_PUBLICATION",\n    "reports/manual_research/study_19.json": "MATCHES_EARLIER_PUBLICATION",\n    "scripts/commodity_close_network.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_feature_diagnosis.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_feature_round.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_first_fold.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_information_audit.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_next_research.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_prior_research.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_public_update.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_publication.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_rank_state_ablation.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_research_release.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_response_encoding.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_session_ablation.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_target_context.py": "MATCHES_CURRENT_PUBLICATION",\n    "scripts/commodity_two_feature_rounds.py": "MATCHES_CURRENT_PUBLICATION",\n    "tests/test_manual_next_research.py": "MATCHES_CURRENT_PUBLICATION",\n    "tests/test_manual_prior_research.py": "MATCHES_CURRENT_PUBLICATION",\n    "tests/test_manual_publication.py": "MATCHES_CURRENT_PUBLICATION"\n  },\n  "patches": {\n    "scripts/commodity_close_network.py": {\n      "input_sha256": "3b0c9b77b5e44962bd9c0068c7f958a7418f826943f5c025aeff2261d4cdf8d4",\n      "patched_sha256": "fbf9c2029e97b212199fe35dda98d9f8cab4b0ce14616e794b150df47698bc71",\n      "edits": [\n        {\n          "old": "w",\n          "new": "_w",\n          "offset": 15055,\n          "reason": "Name intentionally unused iteration value (B007)."\n        }\n      ]\n    },\n    "scripts/commodity_feature_diagnosis.py": {\n      "input_sha256": "a8991b0e9b5517cf5c6c19659084c999aec3cfb7addb28252470b738df5ef80b",\n      "patched_sha256": "f7fe49e7c16c7310d7d835fa6349793633aa85aa87d2a4944dc5257e3db519d8",\n      "edits": [\n        {\n          "old": "name",\n          "new": "_name",\n          "offset": 27396,\n          "reason": "Name intentionally unused iteration value (B007)."\n        }\n      ]\n    },\n    "scripts/commodity_feature_round.py": {\n      "input_sha256": "fc27479ee4821dec7e4f965e1d32c885f7cad0b157b6e1bb3b1cd7ac9822ea73",\n      "patched_sha256": "3c24287cd07665921bfc5138199f5b679cbf402600df049b624d3781efd462a2",\n      "edits": [\n        {\n          "old": "",\n          "new": " from e",\n          "offset": 64559,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": ", strict=False",\n          "offset": 45185,\n          "reason": "Explicit original zip truncation semantics (B905)."\n        },\n        {\n          "old": "l",\n          "new": "low_price",\n          "offset": 26983,\n          "reason": "Unambiguous OHLC variable, unchanged values (E741)."\n        },\n        {\n          "old": "l",\n          "new": "low_price",\n          "offset": 26938,\n          "reason": "Unambiguous OHLC variable, unchanged values (E741)."\n        },\n        {\n          "old": "l",\n          "new": "low_price",\n          "offset": 26924,\n          "reason": "Unambiguous OHLC variable, unchanged values (E741)."\n        },\n        {\n          "old": "l",\n          "new": "low_price",\n          "offset": 26876,\n          "reason": "Unambiguous OHLC variable, unchanged values (E741)."\n        },\n        {\n          "old": "",\n          "new": ", strict=False",\n          "offset": 13151,\n          "reason": "Explicit original zip truncation semantics (B905)."\n        },\n        {\n          "old": "",\n          "new": ", strict=False",\n          "offset": 13029,\n          "reason": "Explicit original zip truncation semantics (B905)."\n        }\n      ]\n    },\n    "scripts/commodity_first_fold.py": {\n      "input_sha256": "517c76064e2c70469843b007c446caf7cc8678440d61c095f42feb6faf62b53d",\n      "patched_sha256": "348ef0a38a62afc239f2f27c31c597ec788fe86911446fe891c2ab0ad1f2be9a",\n      "edits": [\n        {\n          "old": "",\n          "new": " from exc",\n          "offset": 44794,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": ", strict=False",\n          "offset": 1342,\n          "reason": "Explicit original zip truncation semantics (B905)."\n        }\n      ]\n    },\n    "scripts/commodity_information_audit.py": {\n      "input_sha256": "e86f608be525d41c667d261cc0c9fa04a51f08e17b2a58d0a81224a9771d27df",\n      "patched_sha256": "e86f608be525d41c667d261cc0c9fa04a51f08e17b2a58d0a81224a9771d27df",\n      "edits": []\n    },\n    "scripts/commodity_next_research.py": {\n      "input_sha256": "d923cfdbe726d0c068612f5c5066beec4fe631b9919e1425e08de114bcf4dc9d",\n      "patched_sha256": "f650fce14adb6925e843f8c52b16b214c067a008ac3fd1049845cb3d69f313df",\n      "edits": [\n        {\n          "old": "",\n          "new": " from _publication_error",\n          "offset": 24261,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": " as _publication_error",\n          "offset": 24210,\n          "reason": "Name caught exception for explicit chaining."\n        },\n        {\n          "old": "def count(mask):return pd.DataFrame(mask.astype(float)).rolling(w,min_periods=1).sum().to_numpy(copy=True)",\n          "new": "def count(mask, window=w):return pd.DataFrame(mask.astype(float)).rolling(window,min_periods=1).sum().to_numpy(copy=True)",\n          "offset": 6523,\n          "reason": "Bind per-window local rolling helpers explicitly (B023)."\n        },\n        {\n          "old": "def mean(z):return pd.DataFrame(z).rolling(w,min_periods=minimum).mean().to_numpy(copy=True)",\n          "new": "def mean(z, window=w, minimum=minimum):return pd.DataFrame(z).rolling(window,min_periods=minimum).mean().to_numpy(copy=True)",\n          "offset": 6422,\n          "reason": "Bind per-window local rolling helpers explicitly (B023)."\n        }\n      ]\n    },\n    "scripts/commodity_prior_research.py": {\n      "input_sha256": "eb3e968efebf49c036bd42f1610efc33444a7706059030cb18de5773285d0cc8",\n      "patched_sha256": "97195a61b25bcde2ec95785b139a1f57c8192cfecdb1e9142595821a9ec3062b",\n      "edits": [\n        {\n          "old": "",\n          "new": " from exc",\n          "offset": 67599,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": " from _publication_error",\n          "offset": 52732,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": " as _publication_error",\n          "offset": 52646,\n          "reason": "Name caught exception for explicit chaining."\n        },\n        {\n          "old": "gate",\n          "new": "_gate",\n          "offset": 40057,\n          "reason": "Retain call/indexing while naming unused binding (F841)."\n        }\n      ]\n    },\n    "scripts/commodity_public_update.py": {\n      "input_sha256": "3ee316a503f19a8b9e04608f4c8042f6b32840e264beed8bb9681e02948bca7a",\n      "patched_sha256": "a1b77a65cfe767a2012aebf8700b68009df2f11a0dcd271f8d226f00dad65d82",\n      "edits": [\n        {\n          "old": "",\n          "new": " from e",\n          "offset": 50482,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": ", strict=False",\n          "offset": 23429,\n          "reason": "Explicit original zip truncation semantics (B905)."\n        }\n      ]\n    },\n    "scripts/commodity_publication.py": {\n      "input_sha256": "41f41e98fc81de6e0897970537b3751fc40a5dfe32408ccf8661845cdaafe21f",\n      "patched_sha256": "41f41e98fc81de6e0897970537b3751fc40a5dfe32408ccf8661845cdaafe21f",\n      "edits": []\n    },\n    "scripts/commodity_rank_state_ablation.py": {\n      "input_sha256": "bce98b5326fad9fd1e1f04d2536629b1641d8428443e09de4ac641ea87a40762",\n      "patched_sha256": "bce98b5326fad9fd1e1f04d2536629b1641d8428443e09de4ac641ea87a40762",\n      "edits": []\n    },\n    "scripts/commodity_research_release.py": {\n      "input_sha256": "cc8d68e69aa42fedd95a2002f238d1ff9437ce80dbbed2f4b8ca71e1d08f7827",\n      "patched_sha256": "04d52d01eb4439c6f14f0bf3491ff6def6e20399d14cc9247c497198a495413f",\n      "edits": [\n        {\n          "old": "",\n          "new": " from e",\n          "offset": 507657,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": " from _publication_error",\n          "offset": 507361,\n          "reason": "Retain original cause in failure traceback (B904)."\n        },\n        {\n          "old": "",\n          "new": " as _publication_error",\n          "offset": 507294,\n          "reason": "Name caught exception for explicit chaining."\n        }\n      ]\n    },\n    "scripts/commodity_response_encoding.py": {\n      "input_sha256": "51db0fcdc93dc75e671c269ade763797f5c555a10ea7d5b0c22d9fd270f7b496",\n      "patched_sha256": "abe63386847c747e1995b521e38b03605f4163ce8c0581c57c356199ee8108a6",\n      "edits": [\n        {\n          "old": "lag",\n          "new": "_lag",\n          "offset": 11031,\n          "reason": "Name intentionally unused iteration value (B007)."\n        }\n      ]\n    },\n    "scripts/commodity_session_ablation.py": {\n      "input_sha256": "7efef4574d55bd9aad8da8ecf4e7f4ba7cc54907412dcfb4a754100d9e34f88f",\n      "patched_sha256": "7efef4574d55bd9aad8da8ecf4e7f4ba7cc54907412dcfb4a754100d9e34f88f",\n      "edits": []\n    },\n    "scripts/commodity_target_context.py": {\n      "input_sha256": "4c9f2c11f76fe6f9f31516ae078610879fb50bad4b4a55ae0ce3a9ca5bfbd2c3",\n      "patched_sha256": "4c9f2c11f76fe6f9f31516ae078610879fb50bad4b4a55ae0ce3a9ca5bfbd2c3",\n      "edits": []\n    },\n    "scripts/commodity_two_feature_rounds.py": {\n      "input_sha256": "a4284a2a80e42568899704e694b98ef351181be7aa33b3af86135a857a0852aa",\n      "patched_sha256": "8f7742104146df28cd19b053924c08c3be8be516fac87677ac48b18760bec660",\n      "edits": [\n        {\n          "old": "rnd",\n          "new": "_rnd",\n          "offset": 40474,\n          "reason": "Retain call/indexing while naming unused binding (F841)."\n        }\n      ]\n    },\n    "tests/test_manual_next_research.py": {\n      "input_sha256": "a84438ce5508c56de2106291de70184835e6f83032ed26b5a1124cbd098e8430",\n      "patched_sha256": "26871e463ee117f19f05d4ec0fd71c4e3e8b3e6a67e51f0d7008a13fe2a476e2",\n      "edits": [\n        {\n          "old": "import commodity_next_research as m",\n          "new": "m = importlib.import_module(\'commodity_next_research\')",\n          "offset": 455,\n          "reason": "Explicit dynamic import after local path bootstrap (E402)."\n        },\n        {\n          "old": "",\n          "new": "import importlib\\n",\n          "offset": 119,\n          "reason": "Standard-library dynamic import support."\n        }\n      ]\n    },\n    "tests/test_manual_prior_research.py": {\n      "input_sha256": "d5d0475ebac01f3104ac2bbf34dd75fadc47ebced0c086aab545ea9fccabccad",\n      "patched_sha256": "9e4fe6e61cde311424532ef723a0531636ae1dfc2a9166236c8dad4cc663aef7",\n      "edits": [\n        {\n          "old": "q=lambda s:self.a[:,:,self.names.index(\'released_priors__\'+s)].astype(float)",\n          "new": "def q(s):\\n            return self.a[:,:,self.names.index(\'released_priors__\'+s)].astype(float)",\n          "offset": 2765,\n          "reason": "Named local function instead of lambda binding (E731)."\n        },\n        {\n          "old": "import commodity_prior_research as m",\n          "new": "m = importlib.import_module(\'commodity_prior_research\')",\n          "offset": 737,\n          "reason": "Explicit dynamic import after local path bootstrap (E402)."\n        },\n        {\n          "old": "",\n          "new": "import importlib\\n",\n          "offset": 338,\n          "reason": "Standard-library dynamic import support."\n        }\n      ]\n    },\n    "tests/test_manual_publication.py": {\n      "input_sha256": "83f3fbceef4214e8788107655cb826e55bb77199c2605645d1f4ab9c85f1e53b",\n      "patched_sha256": "e43691637319a14a9cd012218b128941b1e9237fb51b77066e0014926a5bca0b",\n      "edits": [\n        {\n          "old": "import commodity_publication as m",\n          "new": "m = importlib.import_module(\'commodity_publication\')",\n          "offset": 278,\n          "reason": "Explicit dynamic import after local path bootstrap (E402)."\n        },\n        {\n          "old": "",\n          "new": "import importlib\\n",\n          "offset": 91,\n          "reason": "Standard-library dynamic import support."\n        }\n      ]\n    }\n  },\n  "quality_diagnostics": {\n    "total": 1754,\n    "file_count": 18\n  }\n}'
)

if __name__ == "__main__":
    raise SystemExit(main())
