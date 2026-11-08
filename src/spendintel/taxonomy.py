"""A 3-level spend taxonomy for a mid-size manufacturer, written for this demo.

The codes are made up (L1 "MR", L2 "MR-FAS", L3 "MR-FAS-BLT"). They follow the shape of public
category trees (segment, family, class) without copying any real code set.

Each leaf carries:
- rule keywords: regex fragments the rules tier uses
- items: what the synthetic plants actually buy, with a unit, a base unit price, and the ways the
  item is written on invoices (abbreviations, part-number style, terse text)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    key: str
    unit: str
    price: float
    texts: tuple[str, ...]  # description templates; {n} is a quantity, {pk} a pack size


@dataclass(frozen=True)
class Leaf:
    code: str
    name: str
    keywords: tuple[str, ...]
    items: tuple[Item, ...]

    @property
    def l1(self) -> str:
        return self.code.split("-")[0]

    @property
    def l2(self) -> str:
        return "-".join(self.code.split("-")[:2])


@dataclass(frozen=True)
class Node:
    code: str
    name: str
    addressable: bool = True
    children: tuple = field(default_factory=tuple)


def _i(key: str, unit: str, price: float, *texts: str) -> Item:
    return Item(key, unit, price, texts)


L1_NAMES = {
    "MR": "MRO and Industrial Supplies",
    "RM": "Raw Materials",
    "PK": "Packaging",
    "LG": "Logistics",
    "IT": "IT and Telecom",
    "PS": "Professional Services",
    "FC": "Facilities",
    "EL": "Electrical and Automation",
}

L2_NAMES = {
    "MR-FAS": "Fasteners",
    "MR-PPE": "Safety and PPE",
    "MR-CUT": "Cutting Tools and Abrasives",
    "MR-LUB": "Lubricants and Fluids",
    "RM-MET": "Metals",
    "RM-PLS": "Plastics and Resins",
    "PK-COR": "Corrugated and Cartons",
    "PK-PAL": "Pallets and Crates",
    "PK-FLM": "Film, Tape and Strapping",
    "LG-FRT": "Freight",
    "LG-WHS": "Warehousing",
    "IT-HW": "Hardware",
    "IT-SW": "Software",
    "IT-TEL": "Telecom",
    "PS-CON": "Consulting",
    "PS-LEG": "Legal and Audit",
    "PS-STF": "Temporary Labor",
    "FC-JAN": "Janitorial",
    "FC-UTL": "Utilities",
    "FC-MNT": "Building Maintenance",
    "EL-CMP": "Electrical Components",
    "EL-AUT": "Automation and Controls",
}

# Utilities are bought on regulated tariffs. A procurement team cannot negotiate them the way it
# negotiates gloves or freight, so they count toward total spend but not addressable spend.
NON_ADDRESSABLE_L2 = {"FC-UTL"}

LEAVES: tuple[Leaf, ...] = (
    # MRO
    Leaf(
        "MR-FAS-BLT",
        "Bolts and Screws",
        (r"\bbolts?\b", r"\bscrews?\b", r"\bcap ?screw", r"\bhex hd\b"),
        (
            _i(
                "bolt-ss-m8x30",
                "ea",
                0.42,
                "stainless steel bolt M8x30",
                "SS bolt M8x30 qty {n}",
                "SS HEX HD BOLT M8 X 30MM",
                "bolt m8x30 a2 stainless",
            ),
            _i(
                "bolt-zn-m10x50",
                "ea",
                0.31,
                "zinc plated hex bolt M10x50",
                "ZN HEX BOLT M10X50 gr 8.8",
                "hex bolt m10 x 50 zinc",
            ),
            _i(
                "screw-sh-m6x20",
                "ea",
                0.18,
                "socket head cap screw M6x20",
                "SHCS M6x20 blk oxide",
                "cap screw m6-20 socket",
            ),
        ),
    ),
    Leaf(
        "MR-FAS-NUT",
        "Nuts and Washers",
        (r"\bnuts?\b", r"\bwashers?\b", r"\block ?nut", r"\bnyloc\b"),
        (
            _i("nut-ss-m8", "ea", 0.11, "stainless hex nut M8", "SS HEX NUT M8 qty {n}", "hex nut m8 a2"),
            _i("washer-flat-m10", "ea", 0.05, "flat washer M10 zinc", "FLAT WSHR M10 ZN", "washer m10 flat"),
            _i("nut-nyloc-m6", "ea", 0.09, "nylon insert lock nut M6", "NYLOC NUT M6", "lock nut m6 nylon ins"),
        ),
    ),
    Leaf(
        "MR-PPE-GLV",
        "Gloves",
        (r"\bgloves?\b", r"\bnitrile\b", r"\bglv\b"),
        (
            _i(
                "glove-nitrile-l",
                "box",
                11.80,
                "nitrile gloves large box/100",
                "nitrile gloves L box/100",
                "NITRILE GLV LG 100CT",
                "disp nitrile glove size L",
            ),
            _i(
                "glove-cut-a4",
                "pair",
                6.40,
                "cut resistant gloves ANSI A4",
                "CUT RES GLOVE A4 sz 9",
                "glove cut lvl a4 pr",
            ),
        ),
    ),
    Leaf(
        "MR-PPE-EYE",
        "Eye and Face Protection",
        (r"\bsafety glasses\b", r"\bgoggles?\b", r"\bface ?shield", r"\beyewear\b"),
        (
            _i(
                "glasses-clear",
                "ea",
                3.90,
                "safety glasses clear lens",
                "SAFETY GLASSES CLR AF",
                "clear safety eyewear",
            ),
            _i(
                "faceshield-8in",
                "ea",
                14.50,
                "face shield 8 inch visor",
                "FACESHIELD 8IN W/ HDGR",
                "face shield visor 8in",
            ),
        ),
    ),
    Leaf(
        "MR-PPE-EAR",
        "Hearing Protection",
        (r"\bear ?plugs?\b", r"\bear ?muffs?\b", r"\bhearing\b"),
        (_i("earplug-foam", "box", 24.00, "foam ear plugs box/200", "EARPLUGS FOAM 200PR", "ear plug foam uncorded"),),
    ),
    Leaf(
        "MR-CUT-DRL",
        "Drill Bits and Taps",
        (r"\bdrill\b", r"\bbits?\b", r"\btaps?\b", r"\bcarbide\b"),
        (
            _i("drill-hss-8mm", "ea", 4.20, "HSS jobber drill bit 8mm", "DRILL HSS 8.0MM JOBBER", "8mm hss drill"),
            _i(
                "drill-carb-10mm",
                "ea",
                38.00,
                "solid carbide drill 10mm",
                "CARB DRILL 10MM 5XD",
                "carbide drill bit 10 mm",
            ),
            _i("tap-m8", "ea", 9.80, "spiral point tap M8x1.25", "TAP M8X1.25 SPPT", "m8 tap spiral pt"),
        ),
    ),
    Leaf(
        "MR-CUT-ABR",
        "Abrasives",
        (r"\bgrinding\b", r"\babrasive", r"\bflap disc", r"\bsanding\b", r"\bcut ?off wheel"),
        (
            _i(
                "disc-flap-4.5",
                "ea",
                5.60,
                "flap disc 4.5 inch 60 grit",
                "FLAP DISC 4-1/2 60G",
                "4.5in flap disc grit 60",
            ),
            _i("wheel-cutoff-6", "ea", 3.10, "cut off wheel 6 inch", "CUTOFF WHL 6X.045", "6in cutoff wheel thin"),
            _i("belt-sand-2x72", "ea", 7.40, "sanding belt 2x72 80 grit", "SAND BELT 2X72 P80", "2x72 belt 80g"),
        ),
    ),
    Leaf(
        "MR-LUB-GRS",
        "Grease and Oil",
        (r"\bgrease\b", r"\bhydraulic oil\b", r"\blubricant", r"\bgear oil\b", r"\blube\b"),
        (
            _i(
                "grease-ep2",
                "tube",
                6.80,
                "EP2 lithium grease cartridge",
                "GREASE EP2 14OZ CART",
                "lithium grease ep-2 tube",
            ),
            _i(
                "oil-hyd-46",
                "pail",
                92.00,
                "hydraulic oil ISO 46 5 gal pail",
                "HYD OIL AW46 5GAL",
                "iso 46 hydraulic oil pail",
            ),
        ),
    ),
    Leaf(
        "MR-LUB-CUT",
        "Cutting Fluids and Coolant",
        (r"\bcoolant\b", r"\bcutting fluid\b", r"\bsoluble oil\b"),
        (
            _i(
                "coolant-semi-55",
                "drum",
                640.00,
                "semi synthetic coolant 55 gal drum",
                "COOLANT SEMI-SYN 55G DR",
                "cutting fluid semisynthetic drum",
            ),
        ),
    ),
    # Raw materials
    Leaf(
        "RM-MET-STL",
        "Carbon Steel",
        (r"\bsteel sheet\b", r"\bhr sheet\b", r"\bcr sheet\b", r"\bsteel plate\b", r"\ba36\b", r"\bcrs\b"),
        (
            _i(
                "steel-crs-16ga",
                "cwt",
                68.00,
                "cold rolled steel sheet 16 ga",
                "CRS SHEET 16GA 48X96",
                "cold roll sht 16 gauge",
            ),
            _i(
                "steel-a36-plate",
                "cwt",
                74.00,
                "A36 steel plate 1/2 inch",
                "A36 PLATE .500 X 96 X 240",
                "hot rolled plate a36 1/2in",
            ),
        ),
    ),
    Leaf(
        "RM-MET-ALU",
        "Aluminum",
        (r"\balumin", r"\b6061\b", r"\b5052\b", r"\balum\b"),
        (
            _i("alu-6061-bar", "lb", 3.40, "6061 T6 aluminum round bar", "AL 6061-T6 RD BAR 2IN", "alum bar 6061 2 in"),
            _i(
                "alu-5052-sheet",
                "lb",
                3.10,
                "5052 H32 aluminum sheet .125",
                "5052-H32 SHT .125 48X144",
                "aluminium sheet 5052",
            ),
        ),
    ),
    Leaf(
        "RM-MET-SST",
        "Stainless Steel",
        (r"\bstainless (sheet|coil|bar|tube)\b", r"\b304\b", r"\b316l?\b"),
        (
            _i("ss-304-coil", "lb", 1.95, "304 stainless coil 2B finish", "SS 304 COIL 2B .060", "304 ss coil 2b"),
            _i("ss-316-tube", "ft", 11.20, "316L stainless tube 1 inch", "316L TUBE 1OD X .065W", "tube 316l 1in od"),
        ),
    ),
    Leaf(
        "RM-PLS-RES",
        "Resins and Pellets",
        (r"\bresin\b", r"\bpellets?\b", r"\bpolypropylene\b", r"\bhdpe\b", r"\babs\b"),
        (
            _i(
                "resin-pp-homo",
                "lb",
                0.92,
                "polypropylene homopolymer resin",
                "PP HOMO RESIN MFI 12",
                "pp pellets homopolymer",
            ),
            _i(
                "resin-hdpe", "lb", 0.88, "HDPE injection grade resin", "HDPE INJ GRADE PELLETS", "hdpe resin injection"
            ),
            _i("resin-abs", "lb", 1.45, "ABS resin natural", "ABS NAT PELLET GP", "abs pellets natural"),
        ),
    ),
    Leaf(
        "RM-PLS-SHT",
        "Plastic Sheet and Rod",
        (r"\bacrylic\b", r"\bpolycarbonate\b", r"\buhmw\b", r"\bdelrin\b", r"\bacetal\b"),
        (
            _i("sheet-uhmw", "ea", 145.00, "UHMW sheet 1/2 x 48 x 96", "UHMW SHT .5X48X96", "uhmw polyethylene sheet"),
            _i("rod-acetal", "ft", 8.60, "acetal rod 2 inch black", "ACETAL ROD 2IN BLK", "delrin rod 2in"),
        ),
    ),
    # Packaging
    Leaf(
        "PK-COR-BOX",
        "Corrugated Boxes",
        (r"\bbox(es)?\b", r"\bcorrugated\b", r"\bcarton", r"\brsc\b", r"\bctn\b"),
        (
            _i(
                "box-rsc-18",
                "ea",
                1.35,
                "corrugated box RSC 18x12x12 32 ECT",
                "RSC 18X12X12 32ECT",
                "carton 18x12x12 single wall",
            ),
            _i(
                "box-rsc-24",
                "ea",
                2.10,
                "corrugated box RSC 24x18x18 44 ECT",
                "RSC 24X18X18 44ECT DW",
                "box 24x18x18 double wall",
            ),
        ),
    ),
    Leaf(
        "PK-COR-INS",
        "Inserts and Dunnage",
        (r"\bdunnage\b", r"\bfoam insert", r"\bpartition", r"\bair pillow", r"\bvoid fill\b"),
        (
            _i(
                "dunnage-pillow",
                "roll",
                38.00,
                "air pillow void fill roll",
                "AIR PILLOW 8X4 ROLL",
                "void fill air pillows",
            ),
            _i("insert-foam", "ea", 0.85, "foam insert custom die cut", "FOAM INSERT DC #4410", "die cut foam insert"),
        ),
    ),
    Leaf(
        "PK-PAL-WOD",
        "Wood Pallets",
        (r"\bpallets?\b", r"\bskids?\b", r"\bgma\b"),
        (
            _i(
                "pallet-48x40", "ea", 14.50, "wood pallet 48x40 GMA grade A", "PALLET 48X40 GMA #1", "48x40 pallet wood"
            ),
            _i(
                "pallet-ht",
                "ea",
                19.00,
                "heat treated export pallet 48x40",
                "PALLET HT ISPM15 48X40",
                "export pallet heat treated",
            ),
        ),
    ),
    Leaf(
        "PK-FLM-STR",
        "Stretch Film",
        (r"\bstretch (film|wrap)\b", r"\bshrink\b", r"\bpallet wrap\b"),
        (
            _i(
                "film-stretch-80",
                "roll",
                26.00,
                "stretch film 18 inch 80 gauge",
                "STRETCH FILM 18IN 80GA 1500FT",
                "pallet wrap 80 ga",
            ),
        ),
    ),
    Leaf(
        "PK-FLM-TAP",
        "Tape and Strapping",
        (r"\btape\b", r"\bstrapping\b", r"\bbanding\b"),
        (
            _i(
                "tape-carton-2",
                "roll",
                2.20,
                "carton sealing tape 2 inch clear",
                "TAPE 2IN X 110YD CLR",
                "packing tape clear 2in",
            ),
            _i(
                "strap-poly-12",
                "coil",
                48.00,
                "polyester strapping 1/2 inch coil",
                "PET STRAP .5IN 6500FT",
                "strapping 1/2 polyester",
            ),
        ),
    ),
    # Logistics
    Leaf(
        "LG-FRT-LTL",
        "LTL Freight",
        (r"\bltl\b", r"\bless than truckload\b", r"\bfreight class\b", r"\bpro ?#"),
        (
            _i(
                "frt-ltl",
                "shipment",
                410.00,
                "LTL freight outbound",
                "LTL FRT PRO# {n}",
                "less than truckload shipment class 70",
            ),
        ),
    ),
    Leaf(
        "LG-FRT-FTL",
        "Truckload Freight",
        (r"\btruckload\b", r"\bftl\b", r"\bdry van\b", r"\bflatbed\b"),
        (_i("frt-ftl", "load", 2350.00, "full truckload dry van", "FTL DRY VAN LANE {n}", "truckload flatbed lane"),),
    ),
    Leaf(
        "LG-FRT-PAR",
        "Parcel and Courier",
        (r"\bparcel\b", r"\bcourier\b", r"\bground ship", r"\bovernight\b", r"\bexpress\b"),
        (_i("frt-parcel", "pkg", 18.50, "parcel ground shipping", "PARCEL GND TRK# {n}", "courier overnight pkg"),),
    ),
    Leaf(
        "LG-WHS-STO",
        "Warehouse Storage",
        (r"\bstorage\b", r"\bpallet positions?\b", r"\bwarehous", r"\b3pl\b"),
        (
            _i(
                "whs-storage",
                "pallet-mo",
                16.00,
                "3PL storage per pallet per month",
                "STORAGE PLT/MO",
                "warehouse storage pallet positions",
            ),
            _i(
                "whs-handling",
                "pallet",
                7.50,
                "3PL handling in and out",
                "HANDLING IN/OUT PER PLT",
                "warehouse handling charge",
            ),
        ),
    ),
    # IT
    Leaf(
        "IT-HW-LAP",
        "Laptops and Desktops",
        (r"\blaptops?\b", r"\bnotebook\b", r"\bdesktops?\b", r"\bworkstation\b"),
        (
            _i(
                "laptop-14",
                "ea",
                1180.00,
                "laptop 14 inch 16GB 512GB",
                "LAPTOP 14IN I7 16GB 512SSD",
                "notebook computer 14in",
            ),
            _i(
                "desktop-sff",
                "ea",
                860.00,
                "desktop small form factor 16GB",
                "DESKTOP SFF 16GB",
                "workstation sff desktop",
            ),
        ),
    ),
    Leaf(
        "IT-HW-PER",
        "Monitors and Peripherals",
        (r"\bmonitors?\b", r"\bdock(ing)?\b", r"\bkeyboard", r"\bmouse\b", r"\bheadset"),
        (
            _i("monitor-27", "ea", 265.00, "27 inch monitor QHD", "MONITOR 27IN QHD USB-C", "27in display monitor"),
            _i("dock-usbc", "ea", 189.00, "USB-C docking station", "DOCK USB-C 90W", "usb c dock station"),
        ),
    ),
    Leaf(
        "IT-SW-LIC",
        "Software Licenses",
        (r"\blicen[cs]e", r"\bsubscription\b", r"\bsaas\b", r"\bseats?\b"),
        (
            _i(
                "sw-cad-seat",
                "seat-yr",
                2400.00,
                "CAD software annual seat",
                "CAD SUBSCR ANNUAL SEAT",
                "cad license 1 yr",
            ),
            _i(
                "sw-office-seat",
                "seat-mo",
                22.00,
                "office suite subscription per user",
                "OFFICE SUITE SEAT/MO",
                "productivity suite license user",
            ),
        ),
    ),
    Leaf(
        "IT-SW-MNT",
        "Software Maintenance",
        (r"\bmaintenance renewal\b", r"\bsupport renewal\b", r"\bsoftware support\b"),
        (
            _i(
                "sw-erp-support",
                "yr",
                18500.00,
                "ERP annual support renewal",
                "ERP SUPPORT RENEWAL FY",
                "software support renewal erp",
            ),
        ),
    ),
    Leaf(
        "IT-TEL-MOB",
        "Mobile Service",
        (r"\bmobile\b", r"\bwireless\b", r"\bcell(ular)?\b", r"\bdata plan\b"),
        (
            _i(
                "tel-mobile-line",
                "line-mo",
                42.00,
                "mobile line monthly service",
                "WIRELESS SVC LINE {n}",
                "cellular data plan monthly",
            ),
        ),
    ),
    Leaf(
        "IT-TEL-NET",
        "Network and Circuits",
        (r"\bcircuit\b", r"\binternet\b", r"\bmpls\b", r"\bfiber\b", r"\bbandwidth\b"),
        (
            _i(
                "tel-circuit",
                "mo",
                1450.00,
                "dedicated internet circuit 1 Gbps",
                "DIA 1G CIRCUIT MRC",
                "fiber internet 1gbps monthly",
            ),
        ),
    ),
    # Professional services
    Leaf(
        "PS-CON-MGT",
        "Management Consulting",
        (r"\bconsult", r"\badvisory\b", r"\bassessment\b"),
        (
            _i(
                "con-ops",
                "day",
                2200.00,
                "operations consulting day rate",
                "CONSULTING SVCS OPS DAYS",
                "advisory engagement ops assessment",
            ),
        ),
    ),
    Leaf(
        "PS-CON-ENG",
        "Engineering Services",
        (r"\bengineering services\b", r"\bdesign services\b", r"\bfea\b", r"\bdrafting\b"),
        (
            _i(
                "eng-hours",
                "hr",
                145.00,
                "contract engineering hours",
                "ENGINEERING SVCS HRS",
                "design engineering drafting hrs",
            ),
        ),
    ),
    Leaf(
        "PS-LEG-LAW",
        "Legal Services",
        (r"\blegal\b", r"\battorney\b", r"\bcounsel\b", r"\blitigation\b"),
        (
            _i(
                "legal-hours",
                "hr",
                420.00,
                "outside counsel legal fees",
                "LEGAL SVCS MATTER {n}",
                "attorney fees litigation",
            ),
        ),
    ),
    Leaf(
        "PS-LEG-AUD",
        "Audit and Tax",
        (r"\baudit\b", r"\btax (prep|services)\b", r"\bfinancial statement"),
        (
            _i(
                "audit-fee",
                "ea",
                9500.00,
                "annual audit progress billing",
                "AUDIT FEE PROGRESS BILL",
                "financial statement audit",
            ),
        ),
    ),
    Leaf(
        "PS-STF-LAB",
        "Temporary Labor",
        (r"\btemp\b", r"\bstaffing\b", r"\bcontract labor\b", r"\bassociate hours\b"),
        (
            _i(
                "temp-labor",
                "hr",
                24.50,
                "temporary labor general associate",
                "TEMP LABOR WK ENDING",
                "staffing associate hours",
            ),
            _i(
                "temp-forklift",
                "hr",
                27.00,
                "temporary forklift operator",
                "TEMP FORKLIFT OPR HRS",
                "contract labor forklift",
            ),
        ),
    ),
    # Facilities
    Leaf(
        "FC-JAN-SUP",
        "Janitorial Supplies",
        (r"\btowels?\b", r"\btissue\b", r"\bcleaner\b", r"\btrash (bag|liner)", r"\bliners?\b", r"\bsoap\b"),
        (
            _i(
                "towel-roll",
                "case",
                42.00,
                "paper towel roll case of 6",
                "TOWEL ROLL 8IN X 800FT CS6",
                "hand towel rolls case",
            ),
            _i(
                "liner-55",
                "case",
                36.00,
                "trash can liner 55 gal case",
                "LINER 38X58 1.5MIL CS100",
                "trash bags 55 gallon",
            ),
            _i(
                "cleaner-deg",
                "gal",
                9.50,
                "degreaser cleaner concentrate",
                "DEGREASER CONC 1GAL",
                "industrial cleaner degreaser",
            ),
        ),
    ),
    Leaf(
        "FC-JAN-SVC",
        "Cleaning Services",
        (r"\bjanitorial services?\b", r"\bcleaning services?\b", r"\bfloor care\b"),
        (
            _i(
                "jan-service",
                "mo",
                6800.00,
                "janitorial services monthly",
                "JANITORIAL SVC MONTHLY",
                "cleaning services contract month",
            ),
        ),
    ),
    Leaf(
        "FC-UTL-ELE",
        "Electricity",
        (r"\belectric(ity)?\b", r"\bkwh\b", r"\bdemand charge\b"),
        (
            _i(
                "util-elec",
                "mwh",
                92.00,
                "electricity usage",
                "ELEC SVC KWH ACCT {n}",
                "electric utility bill demand charge",
            ),
        ),
    ),
    Leaf(
        "FC-UTL-GAS",
        "Natural Gas",
        (r"\bnatural gas\b", r"\btherms?\b", r"\bgas service\b"),
        (_i("util-gas", "dth", 6.20, "natural gas supply", "NAT GAS THERMS ACCT {n}", "gas service therms"),),
    ),
    Leaf(
        "FC-MNT-HVC",
        "HVAC Service",
        (r"\bhvac\b", r"\bchiller\b", r"\bboiler\b", r"\brtu\b", r"\bfilters?\b"),
        (
            _i(
                "hvac-pm",
                "visit",
                950.00,
                "HVAC preventive maintenance visit",
                "HVAC PM QTRLY RTU",
                "hvac service chiller pm",
            ),
            _i("hvac-filter", "ea", 11.00, "HVAC filter 20x25x2 MERV 13", "FILTER 20X25X2 M13", "air filter merv13"),
        ),
    ),
    Leaf(
        "FC-MNT-REP",
        "Building Repairs",
        (r"\broof", r"\bplumbing\b", r"\bdoor repair\b", r"\bdock door\b", r"\bpaving\b"),
        (
            _i(
                "repair-dock-door",
                "ea",
                1350.00,
                "dock door repair service call",
                "DOCK DOOR REPAIR SVC",
                "overhead door repair",
            ),
            _i(
                "repair-plumb",
                "hr",
                115.00,
                "plumbing repair labor",
                "PLUMBING REPAIR LABOR HRS",
                "plumber service call",
            ),
        ),
    ),
    # Electrical
    Leaf(
        "EL-CMP-WIR",
        "Wire and Cable",
        (r"\bwire\b", r"\bcable\b", r"\bawg\b", r"\bso cord\b", r"\bthhn\b"),
        (
            _i("wire-thhn-12", "ft", 0.28, "THHN wire 12 AWG copper", "THHN 12AWG BLK 500FT", "12 awg thhn copper"),
            _i("cable-so-14-4", "ft", 1.65, "SO cord 14/4", "SO CORD 14/4 250FT", "portable cord 14-4"),
        ),
    ),
    Leaf(
        "EL-CMP-MOT",
        "Motors and Drives",
        (r"\bmotors?\b", r"\bvfd\b", r"\bvariable frequency\b", r"\bgearbox\b", r"\bhp\b"),
        (
            _i(
                "motor-5hp",
                "ea",
                690.00,
                "5 HP TEFC motor 1800 rpm",
                "MOTOR 5HP 1800RPM 184T",
                "electric motor 5 hp tefc",
            ),
            _i("vfd-5hp", "ea", 820.00, "variable frequency drive 5 HP", "VFD 5HP 480V", "ac drive 5hp vfd"),
        ),
    ),
    Leaf(
        "EL-CMP-LGT",
        "Lighting",
        (r"\bled\b", r"\blamps?\b", r"\bfixtures?\b", r"\bhigh bay\b", r"\bbulbs?\b"),
        (
            _i(
                "light-highbay",
                "ea",
                165.00,
                "LED high bay fixture 150W",
                "LED HIGHBAY 150W 5000K",
                "high bay led fixture",
            ),
            _i("lamp-t8", "ea", 6.20, "LED T8 tube 4 ft", "LED T8 4FT 18W", "t8 led lamp 4ft"),
        ),
    ),
    Leaf(
        "EL-AUT-SEN",
        "Sensors and Switches",
        (r"\bsensors?\b", r"\bprox(imity)?\b", r"\bphotoeye\b", r"\blimit switch\b", r"\bencoder\b"),
        (
            _i(
                "sensor-prox-m18",
                "ea",
                58.00,
                "inductive proximity sensor M18",
                "PROX SENSOR M18 PNP",
                "m18 inductive prox",
            ),
            _i(
                "photoeye",
                "ea",
                112.00,
                "photoelectric sensor retroreflective",
                "PHOTOEYE RETRO 4M",
                "photo eye sensor",
            ),
        ),
    ),
    Leaf(
        "EL-AUT-PLC",
        "PLCs and Controls",
        (r"\bplc\b", r"\bhmi\b", r"\bi/o module\b", r"\bcontroller\b", r"\brelay\b"),
        (
            _i(
                "plc-io-module",
                "ea",
                410.00,
                "PLC digital input module 16 pt",
                "PLC DI MODULE 16PT 24VDC",
                "i/o module digital input",
            ),
            _i(
                "relay-ice-cube",
                "ea",
                14.00,
                "control relay 24VDC DPDT",
                "RELAY 24VDC DPDT W/BASE",
                "ice cube relay 24v",
            ),
        ),
    ),
)

BY_CODE: dict[str, Leaf] = {leaf.code: leaf for leaf in LEAVES}
L3_CODES: tuple[str, ...] = tuple(leaf.code for leaf in LEAVES)
ITEMS: dict[str, tuple[Item, Leaf]] = {item.key: (item, leaf) for leaf in LEAVES for item in leaf.items}


def l1_of(code: str) -> str:
    return code.split("-")[0]


def l2_of(code: str) -> str:
    return "-".join(code.split("-")[:2])


def name_of(code: str) -> str:
    """Human name for a code at any level."""
    if code in BY_CODE:
        return BY_CODE[code].name
    return L2_NAMES.get(code) or L1_NAMES.get(code) or code


def is_addressable(code: str) -> bool:
    return l2_of(code) not in NON_ADDRESSABLE_L2


def tree() -> list[dict]:
    """The taxonomy as nested dicts for the API and UI."""
    out = []
    for l1, l1_name in L1_NAMES.items():
        kids = []
        for l2, l2_name in L2_NAMES.items():
            if l1_of(l2) != l1:
                continue
            leaves = [{"code": lf.code, "name": lf.name} for lf in LEAVES if lf.l2 == l2]
            kids.append({"code": l2, "name": l2_name, "addressable": l2 not in NON_ADDRESSABLE_L2, "children": leaves})
        out.append({"code": l1, "name": l1_name, "children": kids})
    return out
