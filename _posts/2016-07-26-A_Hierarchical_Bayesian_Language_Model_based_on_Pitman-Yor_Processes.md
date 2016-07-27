---
layout: post
title: A Hierarchical Bayesian Language Model based on Pitman-Yor Processes (HPYLM)
category: Chainer
tags:
- HPYLM
excerpt_separator: <!--more-->
---

## 概要

- [A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/acl2006.pdf) を読んだ
- [A Bayesian Interpretation of Interpolated Kneser-Ney](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf) を読んだ
- C++でHPYLMを実装した

<!--more-->

## はじめに

HPYLMはPitan-Yor過程によるスムージングを行うベイズ階層n-gram言語モデルの一種です。

後で記事にしますが可変長ベイズ階層n-gram言語モデルであるVPYLMとは違いHPYLMはn-gramのオーダーを固定します。

## スムージングとHPYLM

テキストデータが以下の3文とします。

```
she will sing
she will like
he will call
```

この時、たとえば単語列she willに続いてlikeが来る確率$P(like\mid she\ will)$は、she willで始まる文が2つあり、そのうちの1つがshe willに続いてlikeが来ているので、$1$割る$2$で$0.5$となります。

しかしhe willにlikeが続くデータはないため、$P(like\mid he\ will)=0$となります。

このようにデータに出てこないものは全て確率が$0$となってしまうのですが、スムージングと呼ばれる方法を用いると$0$ではない適切な確率を計算できるようになります。



ここでは3-gramなモデルで説明を行います。

つまり、ある単語が生成される確率は、以下のように後ろの2単語のみで決まると仮定したモデルです。


$$
	\begin{align}
		P(dog\mid The\ quick\ brown\ fox\ jumps\ over\ the\ lazy) &= P(dog\mid the\ lazy)\nonumber
	\end{align}\
$$

HPYLMは以下の様な文脈木を考え、単語のことを客、木のノードをレストランと呼びます。

![HPYLM](/images/post/2016-07-27/hpylm.png)

HPYLMではこの文脈木を用いて単語の数をカウントします。

たとえばshe willという単語列の後にlikeという単語が何回来たかをカウントしたい場合、文脈木のルートからwill→sheとレストランをたどり、sheというレストランにlikeという客を追加します。

こうすることでshe willに続いてlikeが1回来たとカウントされます。

同様に上記のデータにあるsingとcallもレストランに追加します（図の黒色の客）

この状態では先程と同じく、he willに続いてlikeが来る回数はheのノードにlikeという客がいないため$0$となります。

そこで代理客（図の白色の客）を親のレストランに送ります。

そうするとhe willの後にlikeが続く回数は依然$0$ですが、willに続いてlikeが来る回数が$1$になります。

よって$P(sing\mid he\ will)$を求めるときに、$c(sing\mid he\ will)$と$c(sing\mid will)$をうまく補完すれば$0$ではない値にすることが可能になります。（$c(\cdot)$は客数を表します）

これがスムージングの考え方で、HPYLMではPitman-Yor過程と呼ばれる確率過程を用いて補完しています。

## 学習

HPYLMにおけるパラメータは文脈木内の代理客を含めた全ての客の配置です。

全ての客の配置を$\boldsymbol\Theta$とすると、文脈$\boldsymbol u$に単語$w$が続く確率$P(w\mid\boldsymbol u)$は

$$
	\begin{align}
		P(w\mid\boldsymbol u)=P(w\mid\boldsymbol u, \boldsymbol\Theta)
	\end{align}\
$$

となり、$\boldsymbol\Theta$によって決まります。

我々の目標は$\boldsymbol\Theta$が真の分布からのサンプリング結果になることです。

もしそうなれば式(6)で与えられるn-gram確率は正確な値になるからです。

しかし当然ながら真の分布から直接$\boldsymbol\Theta$をサンプリングすることは難しいので（そもそもここでの真の分布がどのようなものなのか想像ができません）、ギブスサンプリングを用います。

まず${\rm RemoveCustomer(\boldsymbol u,w)}$によりレストラン$\boldsymbol u$から$w$を削除します。

削除された後の残りの客全ての配置を$\lnot\boldsymbol\Theta$とし、$\lnot\boldsymbol\Theta$のもとで$w$の配置を再サンプリングします（${\rm AddCustomer}(\boldsymbol u, w)$）。

こうすると新たな配置$\boldsymbol\Theta^{new}$がギブスサンプリングされたことになります。

以上の操作をランダムに選んだ訓練データを使って繰り返し行うことで、得られた$\boldsymbol\Theta$は真の分布からのサンプリング結果に近づきます。


## 実装

C++での実装を[GitHub](https://github.com/musyoku/hpylm)に上げておきました。

ここからはHPYLMで単語2-gram言語モデルを学習させる前提で実装について説明します。

また用いる記号については論文に合わせて

- $w$
	- 単語
- $\boldsymbol u$
	- 単語$w$の左側にあるすべての単語列（文脈）
- $\pi(\boldsymbol u)$
	- 文脈$\boldsymbol u$のオーダーを1つ下げた文脈
- $\mid\boldsymbol u\mid$
	- 文脈$\boldsymbol u$に含まれる単語数
- $c_{\boldsymbol uwk}$
	- レストラン$\boldsymbol u$で単語$w$を提供しているいくつかのテーブルのうち、$k$番目のテーブルにいる客数
- $c_{\boldsymbol uw\cdot}$
	- レストラン$\boldsymbol u$で単語$w$を提供しているすべてのテーブルの客の総数
- $c_{\boldsymbol u\cdot\cdot}$
	- レストラン$\boldsymbol u$にいる客の総数
- $t_{\boldsymbol uw}$
	- レストラン$\boldsymbol u$で単語$w$を提供しているテーブルの総数
- $t_{\boldsymbol u\cdot}$
	- レストラン$\boldsymbol u$のテーブルの総数
- $d_{\mid\boldsymbol u\mid}$
	- Pitman-Yor過程のハイパーパラメータ
	- レストランごとではなく深さごとに共通の値を使う
- $\theta_{\mid\boldsymbol u\mid}$
	- Pitman-Yor過程のハイパーパラメータ
	- レストランごとではなく深さごとに共通の値を使う
- $G_0(w)$
	- 基底測度（単語0-gram確率）
	- 語彙数の逆数をパラメータに持つ一様分布とする

とします。

たとえば文が$she\ will\ sing$で$w$が$sing$である場合、

$$
	\begin{align}
		w &= sing\nonumber\\
		\boldsymbol u &= she\ will\nonumber\\
		\pi(\boldsymbol u) &= will\nonumber\\
		\mid\boldsymbol u\mid &= 2\nonumber\\
	\end{align}\
$$

となります。

### WordProbability($\boldsymbol u$, $w$)

文脈$\boldsymbol u$の後に$w$が続く確率は

$$
	\begin{align}
		{\rm WordProbability}(\boldsymbol u, w)&=\nonumber\\
		P(w\mid \boldsymbol u, $\boldsymbol\Theta$)&=\frac{c_{\boldsymbol uw\cdot} - d_{\mid\boldsymbol u\mid}t_{\boldsymbol uw}}{\theta_{\mid\boldsymbol u\mid}+c_{\boldsymbol u\cdot\cdot}}
		+\frac{\theta_{\mid\boldsymbol u\mid}+d_{\mid\boldsymbol u\mid}t_{\boldsymbol u\cdot}}{\theta_{\mid\boldsymbol u\mid}+c_{\boldsymbol u\cdot\cdot}}{\rm WordProbability}(\pi(\boldsymbol u), w)\nonumber
	\end{align}\
$$

となり、再帰的に文脈のオーダーを落として計算します。


### AddCustomer($\boldsymbol u$, $w$)

文脈$\boldsymbol u$のもとで単語$w$が観測された時、深さ1に対応するノード（HPYLMでは常にn-1の深さのノードに追加する）に$w$を追加します。

追加する際は、

- $max(0, c_{\boldsymbol uwk} - d_{\mid\boldsymbol u\mid})$に比例する確率で、$w$を提供しているすべてのテーブルの中の$k$番目のテーブルに追加
- $(\theta_{\boldsymbol u} + d_{\boldsymbol u})P(w \mid \pi(\boldsymbol u))$に比例する確率で、$w$を提供する新しいテーブルを作成しそこに追加
	- この時親レストラン$\pi(\boldsymbol u)$に対し${\rm AddCustomer}(\pi(\boldsymbol u), w)$
	- したがって親レストランでも同様に、ある確率でさらにその親に対して${\rm AddCustomer}(\pi(\pi(\boldsymbol u)))$

のように確率的に客をテーブルに追加します。

### RemoveCustomer($\boldsymbol u$, $w$)

客を削除する際は、$c_{\boldsymbol uwk}$に比例する確率で、$w$を提供しているテーブルの中の$k$番目のテーブルから客を削除します。

$k$番目のテーブルに客が一人もいなくなった場合はそのテーブルを削除し、親レストラン$\pi(\boldsymbol u)$に対し${\rm RemoveCustomer}(\pi(\boldsymbol u), w)$を実行します。

### ハイパーパラメータの推定

$\theta_{\mid\boldsymbol u\mid}$や$d_{\mid\boldsymbol u\mid}$は初めに適当な初期値を与えておきますが、現在の文脈木のパラメータからサンプリングにより推定します。

[論文](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf)のAppendix Cに詳細が書いてありますが、まだ理解が追いついていないのでやり方だけ書いておきます。

まず以下の3つの補助変数を定義します。

$$
	\begin{align}
		x_{\boldsymbol u} &\sim {\rm Beta}(\theta_{\mid\boldsymbol u\mid}+1, c_{\boldsymbol u\cdot\cdot}-1)\\
		y_{\boldsymbol ui} &\sim {\rm Bernoulli}\left(\frac{\theta_{\mid\boldsymbol u\mid}}{\theta_{\mid\boldsymbol u\mid} + d_{\mid\boldsymbol u\mid}i}\right)\\
		z_{\boldsymbol uwkj} &\sim {\rm Bernoulli}\left(\frac{j-1}{j-d_{\mid\boldsymbol u\mid}}\right)
	\end{align}\
$$

ベータ分布・ガンマ分布からサンプリングします。

$$
	\begin{align}
		d_m &\sim {\rm Beta}\left(
		a_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}\sum_{i=1}^{t_{\boldsymbol u\cdot - 1}}(1 - y_{\boldsymbol ui}),
		b_m + \sum_{\boldsymbol u,w,k:\mid u \mid=m,c_{\boldsymbol uwk}\geq2}\sum_{j=1}^{c_{\boldsymbol uwk - 1}}(1 - z_{\boldsymbol uwkj})
		\right)\\
		\theta_m &\sim {\rm Gamma}\left(
		\alpha_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}\sum_{i=1}^{t_{\boldsymbol u\cdot - 1}}y_{\boldsymbol ui},
		\beta_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}{\rm log}x_{\boldsymbol u}
		\right)\\
	\end{align}\
$$

$\sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}$は深さ$m$であり、かつ$t_{\boldsymbol u\cdot}\geq2$となるような全てのレストランに対する総和です。

その他の$\sum$についても同様に考えます。

また新たなハイパーパラメータ$a_m, b_m, \alpha_m, \beta_m$が出てきますが、これは適当な値を設定して固定します。

またガンマ分布からのサンプリングの実装には注意が必要です。

[英語版のウィキペディア](https://en.wikipedia.org/wiki/Gamma_distribution)に書いてありますが、ガンマ分布は${\rm Gamma}(k, \theta)$と${\rm Gamma}(\alpha, \beta)$の2種類の表記があり、計算方法が違います。

C++の[gamma_distribution](http://www.cplusplus.com/reference/random/gamma_distribution/)関数は${\rm Gamma}(k, \theta)$の方で実装されているため、この関数を用いる場合は式(5)の２つ目のパラメータ$$\beta_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}{\rm log}x_{\boldsymbol u}$$の逆数をgamma_distributionの2つ目の引数とします。

初めはこの事に気づかず実装したため$\theta_m$の値が数千〜数万という巨大な値になりました。

正しく実装すると小さな実数になります。

## 実験

[Alice in Wonderland](https://www.cs.cmu.edu/~rgs/alice-table.html)の全文に対し文字3-gramを学習させてみました。

学習は数分で終了し、パープレキシティは16に収束しました。

この値は小さすぎる気がするのですが文字n-gramの文字あたりパープレキシティなので妥当な値だと思います。

また学習後のHPYLMを使って文章生成を行わせてみた結果が以下になります。

```
^`Thewidedrumblon'tsheQueedputwayboure!$
^Thehatyedhingaingthemptheshesile,loofit?'$
^`Icapedtoncecursedarknoledthosedtofcoughhen,ashenrelfbecandthenitvebeweretmedtherlyawhandence,'saidAlidAlice,wit,asthmenowlikeaceattiss,metsfitwayanding,'sticeofthecindthalmyprisnoseywelfandwhitwanyoulalowishembefin.$
^'AnderyoustaninheFookat.$
^Shealite.$
^`Aheselicensionbeatyou!'saidn'sain,therpillarringrionstIst,'sadytooke-wedgurentonheKingmestas,'tyonsomewelfbearedall!'wheseldbechandnot.$
^Alice,`Comakeanxionlygod-bothisheMockTurchinghtheHathalneoffitwrou'remak.$
^Andknithessain:itakingaidassordshebefitsnotventhhestreonerackabbittheDorriekneafacestallosomeofthent,'trawnatAlit,Billvoinagolefitprehingh;andfirseybythes,youtonahsualloorsewhatthebriedow!'crientwitactund.$
^Thiskyoflettheredourebegoityoustrys,`ance,ithasshoarkthehekinklikedtoheacuthatesanylice.$
^Sosegasshuterwasaidthepooteisvoinagoonlyinsweloseed,andthecongunin.$
^`Youshed:--$
^Samefirhewasveandthert.'$
^Hatwametin,dows.$
^Thent?'sanconewashchHarldthered`Com,yalareadveterrymakrouse--dalltheOwlarcheHargueendsheQueedrupbytheHatcouldsaywriptesookinclouldopinecoorstbeganneesperyawoudimaccerthethepandmadnowither,thgrepenincerpokeycouttormrouthesaide,thershowedheswer,andourniblow.'$
^Oh,'saidaLorIeaugotherenttingingought?'say-fiethecar.`IsheGry-foundy,$
^Butwasshyoubtfingwhameatdesairome`Thers,anclestwalovealabodnothedthadearitinthoftedheMockleatdinboutheraboujusir,shanthewasce,whingothes,sheth:`forhattudon!$
^Gookehondthetryup,astay.$
^Theforfingo,and,shristhandirlytofherteRabodat'snouknowIsheyhumadfeveryarchaty$
^SoAlicem,'tyelf!'$
^Lon,soruseverseandoverses,'tsiont.`Bectlesaid,whinahme?$
^`WhisorearowningtheKindsou!$
^It'saiddloffforyint,'heGryphoulace.$
^`Comedingototmartwon.$
^`"Why,$
^`Cats$
^Howenewhes,andsoutterwagoospiledullandAlingandththeh,'Alitseerewasswer.$
^`AbacewasshebothecatshoodoidAlikeare,soopeplyI'dreagotbutfuldglicesherherprou'reasuchandsheteRabothellonchou'resayboulls,ordhandwhavantshemakedthichalactleinerepossidtofwhin.$
^TheDidthefor,shoagabitseitwor?'$
^Fat"Isher.$
^`Youndeepasaideextvouseddonce.`Whad!$
^Iwasioselbeaves.$
^`Yeslybothourse:andiddly:thentherwisessidAlit'sitwogehance.$
^"Ugh,stawaskeanwerwhatisonitheyou're.$
^Thesoorlowthenly:`Itogreging?'$
^Aling,$
^Iwite--`Thestyisdon-andmail,fongaidegamealovere?'said,andIhattle.$
^`Youstomeadawfuldithousetion,'sheryhatchistagaingthitshemerseemalwassheptamengingintoftedgoorchshrouwes,andonglosculcaugredgrepingbutsotassheQueenewasmometopproffethethada,doftemesaimeashesaidthePig,'thopentheQueeinger,andeencestgofthtAlice.`Ihance.$
^"Theintingoisize:hetohatthesheryAliced;andbeasthouldyamingthestalkinghthememwinavellthersetonstedtheMarlin.$
^'Alitmallastily,anatheMocouldde,becutformose,bestisedallttemblenIwortletotmayundIhavere,oll,andthinkyouhandnowassupooknofflachfumblearedgrygazinthisheRab,hushargo.$
^`Itwan'tlyadbecatrierheditifsoldne,alktooressshreaverottlithisherwand,pencherwalkinks,plieton'tthal'sainandthePandpaidanheCatthoundenother-fingofuposoryintsuchHaretthEdwentimingthesuchon,AlikedthoicedswassixplawmidthoresairflabbithenIwitactimproulicehatheryinwileshoppersleepord,)`Inalliverhadfuldgame!$
^`Hadfrither)`Astrundwhyoutthine,forhasclandandredwhere,'sainaveofhisthetlybelltthenthice,therfexeclosho'conmyfandit?'saidAlichisponfuldalittowliceleplesthohdentooneadboucarksergother.$
^Howwithatsidglarkedallneaningdothiskmet,andbefuldtoAllf,yousedtasondsaid,'sareingtorey?$
^Ifisherflyseditwatveofeberiongedonturtyontodupackthisbabou,andaysmarthellone,and,begobechavereyetryphoughthilleadtohectledtreandshehofferseyes,anwallieduptohat"ithetheMoughtoneofarpeafanytoftheheMoughisnownheiniso,begatinarcopeedglarge,whingargupsibbinnitandhapincied,`ifthemill:ingthemadnotmomecrouardal,bythattohecladyouwenjuryphon'theesque.'Soutirchongtot!'saveonowntoheyer'sgettagandwhing-dallcri
^'Thingbuttildreare,ancealls.'$
^`Offenlyrot,aswhinthallbehatdonelice.$
^Theyoutiong,theGryit,sonsthehemblassheingaiddinheple,andsaidveritertaMar.$
^`Yousaidtrepy'rewasheFoory.$
^`Theher.`Sevenexpereplice,(ithaterost,shIhebeothen'toggs!$
^`Whathet-ofprigerinasellwroudo.'$
^`Oncesperoolicebeirrierhehekingabbitmally,`Isheonttledomingquingthesomarkeditspefrondbeoftere,asursaidit!$
^`Ihatdofthehangrandwherion.'$
^Fookeditch.$
^I'msen.`Itillofamposee,'Andcringht,aswereedtuou'reselayaboughtitwit;thingundjoingqueenchtAlinisfitindlyesioul,IspentheQuee,$
^NowereadeatimhisvenesehatersthenAlikeeneaddongivehin,nowlargother.$
^`JusellingthatIknoic.'$
^TheMoun?'$
^Firs,'said,ve?$
^Itock,'thesshourforemenconevengothercingin,an,whadeelanyminexecilysewhy,'sagavenddenevery.$
^Soutitelwried;`Ionrowdowoureardleshansibbitnexecom!'coorpagandmarearearcheygriedthalldesedowcutlesn'tther,theis?$
^ForgetthuninkIshey'requedles,thhapedhatton'theadeasarsursetearethenIgrundheverself.'$
^`Wely,thempandvandtheadeoundionevehage,IhatbesurtoandtheebackTureen-andforthempoked,Fat,'heatthinght,'tbeforatobele.$
^Eit,'satholdifyouar,sirebeitwasoleewheHatwaying.$
^"Therforstoldoeasheastitthan"Then'ttoh!'sadres:naminasoffbepin-upagoesuchininketheysrewast,'sandhadvoicetreardlytherivelf.`Isuchad--nesit'sheKingtherightheallanday`Wow,'tmakeyenthestrolarmomEnguppene:anxiouthittimeundthemusitshimeardle,'tashenesaidtheyoutticer!$
^`Onew)tillsaidtonexttheQuallarking,annekeyatfor.$
^'Aliced;`vereliedoesublay!'snicewanEacht,andidAlitteelinnowers$
^Thisherchewen,wingaidyethoutherQuesaidtoldknothejuserarstorrusthewossogont,'saidthimsoresonthereir!$
^Afthatleagedquesteorepillsh).`Oh,andheadnowsalanandtakathealookedifthitheyoubutasseerearthadn'tsoneottleepiesair:shelplanthetwonthowasgon!'saidthejusencuttertleshargundmaduptithedtogothed,pokeswhoeir!"Iearchattersuldmusle,ifyoudednothewtaleithincea-picelitthelfandventlonedbyhory;`Iknossheyefingow.'$
^`Chersedanthebowoncewshettheactallaythad!'sn'tknothavysyougharchoughthefouldoftnothattoh,'tmay.$
^`DRIGHTFOOT,bell!'$
^Alikeot,'saidAlittewitscoughalnextmadeatselicewasbeveberthis,thatslyuserawfusthefed.`Whatyoureoremildgoblestarculdit.'$
^Thelasaindhedtouttenthewit,inghatsmadfoutthenersesapillmuckTurpedtogrowntsfeey,`Yous--'saidthinueereaskedoicurightAlicellainclawhiggisforhesughtatimingrend--evid,sheme,'Alicewaylit:thersons,$
^Youpboddoothonsherher!"'$
^`BrieverwitersandFatenheonerin,`youall,whingladdeconabotamthaddidAliketakenscourycutwingaidAliswoome.$
^'And,`--`Catheedoftenstopepeathryoutht,'Anymuccidly,offbacked.$
^ShentobabbisstheMoulicesaidtohapeoldnismoumannotheLortliedan'ttesurnownbegrosame!$
^TherealindAlintakgoinhathesaiderismallasuldlovem-inamoselitinser,youtelthense,(louldtherety--'$
^SoyoundIhaticearoledtoeafregeoffrundwamightfusetvannityaw,andsidthiseencarookupbythewhyoumshersheWhandthiner.$
^Butintedtofthhit:nallon'tgremeofmignerandsomertyourtardeninawasitheEacakeakelaslowashatdoveyalk.$
^Alice.$
^Buttly:$
^`Whadyelficepallthehurieverecandsheofbecutockitwitshas,anremanothrofhingoaddearieve!'ttlethes,anbeandsofaccingdoodry-ther,afterseliteasther!$
^Assheringortaroquitgoodoesthavetried,rew)toheatsofthediedidthesioldIdafthefer,'sanyounce:`"--shegookincehadeadryinalice.$
^Alicetofnotandedre!$
^`Catpeatog,sherall?'tingtofings:`Oneday.$
^`Thight.$
^`Inasaidnouldscoultaideday,andhoracutsou'reminerem,)`Unitery,iffaucounkIcanbyvotsherswevetird--firsetobsts,asthepingingto--'$
^Bytheliked,asthindmetitgodogreeknobelfsinstinerenerubjelfhatsainknow,'saidAlit?'sditheyelit,ink,ascraboodshavoughthellwropturtandthisomesheentedgolieveoneing.'$
^`I'velice,an--MagargesnotheflastchmeausomedwaseIcourson'tmadnotmesucallshertheiteiremsthecatgothanekingot,whofecould,buttrytheKing,anagandsthewass,andmareacutheGry,'tmalasstonarree"butshesually:`shurlyswelesserienthe'samusen$
^Thattet,anditer!$
^Hown.$
^`IndAlietrePookforherestbacontodthepthadereyeabygeindIcalesandDeartooking,andup!theMar.$
^WoutionheGryonoulthecou'roplatienIgoitapsn'tgroul,`Irephoughtherit'salice,'shebattlearquittlertloverewedoggivegofheare,whingsIworthfloomwhingthinkedthatfalleendwhyle,'sandealkitohine,an'trehelitwouldbehaspouragavitingtion,sheinghtterwitakgotlebouedbowtreallongeotgofeen.'$
^Thelldyleencefor$
^`I'lly.$
^'Aliketoolet,andsobestobebyetoherwitstknownhumyhalookedre.$
^`Catgoorthemedaceditshers,mydoftheKinah'sairsomehed?$
^Willcouthentosametlethime,shenerestionasfitwaidthecondeof-themportsmovedowtriedo,andtheingookehadyouhoughtfusetoratofsoftheofthreanddours--ficeachiralondearecame,whenshecouteance,hebythemetis;buttert,it,anytobbirswen,forntakedtheDoremonwasthtAlice'saideadflaswhicetheir,anhebadouldre,sheMockTurryquar.$
^Hered.$
^`Thes,Whandtoorgetheknotte,theitmustchisblesheoputtersitpurfusturrytherneingithenattaGrythefauchhilesupist,`argethitnotas,thtneitthery--youregasamomsomosen,thadde.'$
^`Thily.$
```
文字3-gramではさすがに小さすぎたのか、いい文章が生成されませんでした。

## 関連

- Pitman-Yor 過程に基づく可変長 n-gram 言語モデル（準備中）
- ベイズ階層言語モデルによる教師なし形態素解析（準備中）
- 隠れセミマルコフモデルに基づく教師なし完全形態素解析（準備中）

## おわりに

初めてDeepではない自然言語処理をやったので、書いたコードが正しいのかどうかわかりません。

