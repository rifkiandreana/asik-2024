// Inisialisasi peta
const map = L.map('mapid').setView([-5.14, 119.42], 5);

// Layer dasar (background peta)
const baseLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
});
baseLayer.addTo(map);


// Layer untuk peta titik budidaya
const titikLayer = L.layerGroup();

// Data GeoJSON
const geojsonData = {
    "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [119.5000, -5.5158]
      },
      "properties": {
        "name": "Kabupaten Takalar",
        "province": "Sulawesi Selatan",
        "penghasil": "Tinggi"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [119.7373, -5.6707]
      },
      "properties": {
        "name": "Kabupaten Jeneponto",
        "province": "Sulawesi Selatan",
        "penghasil": "Tinggi"
      }
    },
    {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [120.5030, -6.2014]
        },
        "properties": {
          "name": "Kabupaten Kepulauan Selayar",
          "province": "Sulawesi Selatan",
          "penghasil": "Tinggi"
        }
      },
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [119.9500, -5.5333]
      },
      "properties": {
        "name": "Kabupaten Bantaeng",
        "province": "Sulawesi Selatan",
        "penghasil": "Tinggi"
      }
    },
    {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [120.2500, -5.1167]
        },
        "properties": {
          "name": "Kabupaten Sinjai",
          "province": "Sulawesi Selatan",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [116.5500, -8.5333]
        },
        "properties": {
          "name": "Kabupaten Lombok Timur",
          "province": "Nusa Tenggara Barat",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [117.4333, -8.4667]
        },
        "properties": {
          "name": "Kabupaten Sumbawa",
          "province": "Nusa Tenggara Barat",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [118.4500, -8.5500]
        },
        "properties": {
          "name": "Kabupaten Dompu",
          "province": "Nusa Tenggara Barat",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [123.6000, -10.1667]
        },
        "properties": {
          "name": "Kabupaten Kupang",
          "province": "Nusa Tenggara Timur",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [123.0667, -10.7333]
        },
        "properties": {
          "name": "Kabupaten Rote Ndao",
          "province": "Nusa Tenggara Timur",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [124.5667, -8.2000]
        },
        "properties": {
          "name": "Kabupaten Alor",
          "province": "Nusa Tenggara Timur",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [128.9333, -3.3667]
        },
        "properties": {
          "name": "Kabupaten Maluku Tengah",
          "province": "Maluku",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [128.3667, -3.0667]
        },
        "properties": {
          "name": "Kabupaten Seram Bagian Barat",
          "province": "Maluku",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [132.7333, -5.5833]
        },
        "properties": {
          "name": "Kabupaten Maluku Tenggara",
          "province": "Maluku",
          "penghasil": "Tinggi"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [113.8667, -7.0000]
        },
        "properties": {
          "name": "Kabupaten Sumenep",
          "province": "Jawa Timur",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [114.0667, -7.7000]
        },
        "properties": {
          "name": "Kabupaten Situbondo",
          "province": "Jawa Timur",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [112.9000, -7.6500]
        },
        "properties": {
          "name": "Kabupaten Pasuruan",
          "province": "Jawa Timur",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [106.2845, -5.9713]
        },
        "properties": {
          "name": "Kabupaten Serang",
          "province": "Banten",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [105.7667, -6.3000]
        },
        "properties": {
          "name": "Kabupaten Pandeglang",
          "province": "Banten",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [97.9683, 4.4721]
        },
        "properties": {
          "name": "Langsa",
          "province": "Aceh",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [97.1703, 3.2790]
        },
        "properties": {
          "name": "Langsa",
          "province": "Tapaktuan",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [95.3175,  5.8839]
        },
        "properties": {
          "name": "Langsa",
          "province": "Sabang",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [ 104.1089, 0.8022]
        },
        "properties": {
          "name": "Pulau Sugi",
          "province": "Riau",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [103.4357,  0.7586]
        },
        "properties": {
          "name": "Pulau Kudur",
          "province": "Riau",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [103.4274, 0.9602]
        },
        "properties": {
          "name": "Pulau Karimun Tinggi",
          "province": "Riau",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [110.3100, -6.9511]
        },
        "properties": {
          "name": "Semarang",
          "province": "Jawa Tengah",
          "penghasil": "Sedang"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [123.8902, -5.6326]
        },
        "properties": {
          "name": "Kabupaten Wakatobi",
          "province": "Sulawesi Tenggara",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [124.8333, 1.4667]
        },
        "properties": {
          "name": "Manado",
          "province": "Sulawesi Utara",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [130.7760,  -0.4200]
        },
        "properties": {
          "name": "Kabupaten Raja Ampat",
          "province": "Papua Barat",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [ 131.2500,  -0.8760]
        },
        "properties": {
          "name": "Kabupaten Sorong",
          "province": "Papua Barat",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [140.7167, -2.5167]
        },
        "properties": {
          "name": "Papua",
          "province": "Papua",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [117.4945, 0.1324]
        },
        "properties": {
          "name": "Kota Bontang",
          "province": "Kalimantan Timur",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [117.9996, 1.1135]
        },
        "properties": {
          "name": "Kutai Timur",
          "province": "Kalimantan Timur",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [114.4167, -3.2000]
        },
        "properties": {
          "name": "Barito Kuala",
          "province": "Kalimantan Selatan",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [109.3333, 0.0500]
        },
        "properties": {
          "name": "Kalimantan Barat",
          "province": "Kalimantan Barat",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [123.0500, 0.5333]
        },
        "properties": {
          "name": "Gorontalo",
          "province": "Gorontalo",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [102.2333, -3.8000]
        },
        "properties": {
          "name": "Bengkulu",
          "province": "Bengkulu",
          "penghasil": "Rendah"
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [105.2667, -5.4167]
        },
        "properties": {
          "name": "Lampung",
          "province": "Lampung",
          "penghasil": "Rendah"
        }
      }
    ]
};


// Fungsi untuk menentukan warna marker berdasarkan tingkat potensi
function getColor(penghasil) {
    switch (penghasil) {
        case "Tinggi":
            return "green"; // Potensi tinggi
        case "Sedang":
            return "yellow"; // Potensi sedang
        case "Rendah":
            return "red"; // Potensi Rendah
        default:
            return "gray"; // Default
    }
}

// Tambahkan data GeoJSON ke layer titik budidaya
L.geoJSON(geojsonData, {
    pointToLayer: (feature, latlng) => {
        // Tambahkan marker dengan warna berdasarkan tingkat potensi
        const color = getColor(feature.properties.penghasil);
        return L.circleMarker(latlng, {
            radius: 8,
            fillColor: color,
            color: "#000",
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        });
        
    },
    
    onEachFeature: (feature, layer) => {
        // Tambahkan popup dengan informasi properti
        const { name, province, penghasil } = feature.properties;
        layer.bindPopup(`
            <strong>${name}</strong><br>
            Provinsi: ${province}<br>
            Penghasil: ${penghasil}
        `);
    }
}).addTo(titikLayer);

/// Menambahkan legenda pada layer
const legend = L.control({ position: "topright" });

legend.onAdd = function () {
    const div = L.DomUtil.create('div', 'info legend card');
    const grades = ['Tinggi', 'Sedang', 'Rendah']; // Kategori potensi
    const colors = ['green', 'yellow', 'red']; // Warna untuk masing-masing kategori

    // Header legenda
    div.innerHTML = `<div class="legend-header"><strong>Penghasil Rumput Laut</strong></div>`;

    // Menyimpan elemen HTML untuk tiap kategori
    for (let i = 0; i < grades.length; i++) {
        div.innerHTML += `
            <div class="legend-item" data-category="${grades[i]}">
                <i class="legend-icon" style="background-color:${colors[i]}"></i>
                ${grades[i]}
            </div>
        `;
    }

    return div;
};



// Menambahkan legenda ke peta
legend.addTo(map);

// Kontrol layer
//L.control.layers(null, { "Peta Titik Budidaya": titikLayer }, { collapsed: false }).addTo(map);

// Menampilkan layer titik budidaya sebagai layer default
titikLayer.addTo(map);

