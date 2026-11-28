# Grafana LK-11

[Dashboard JSON](dashboards/lk-11-operational-monitoring.json) memuat `MLOps-Electricity-Demand-Forecasting - Dashboard`. File tersebut mempertahankan UID dashboard yang diekspor dari Grafana dan memakai datasource `prometheus` dari [provisioning datasource](provisioning/datasources/prometheus.yml).

[Provider dashboard](provisioning/dashboards/lk-11.yaml) memuat JSON ke folder General. Perubahan panel disimpan di JSON, lalu Grafana membaca ulang direktori dashboard setiap 30 detik.

Untuk memuat provisioning lokal dari root repository:

```powershell
docker compose config -q
docker compose up -d --no-deps grafana
```

Buka [dashboard LK-11](http://127.0.0.1:3000/d/792ce94f-4dfa-4c12-8fb9-ff28d0063d14/mlops-electricity-demand-forecasting-dashboard). Enam panel Stat membaca metrik `ciso_production_*` melalui Prometheus. Prometheus memakai target `production-metrics:9108` pada job `ciso-production`. [Dokumentasi provisioning Grafana](https://grafana.com/docs/grafana/latest/administration/provisioning/#dashboards)
