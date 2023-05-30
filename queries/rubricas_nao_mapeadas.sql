select 
    tr.cod
    ,tr.DESCR_RUBRICA as nome_rubrica
from BD_PESSOAL_EXTRA_SIAPE.dbo.TIPO_RUBRICA tr,
    bdu_sefip.dbo.orgao_extra_siape_clientela osc
where not exists (
    select 1  from bdu_sefip.dbo.extra_siape_tipo_rubrica_mapeada trm
    where tr.cod = trm.cod
    )
and tr.COD_ORGAO = osc.cod_clientela
--UNION 
--select trs.nome_rubrica as nome_rubrica
--from BD_PESSOAL_SIAPE_AMPLO.dbo.TIPO_RUBRICA trs
--where not exists (
    --select 1  from bdu_sefip.dbo.siape_tipo_rubrica_mapeada trsm
    --where trs.cod_rubrica  = trsm.cod 
    --)
--order by nome_rubrica