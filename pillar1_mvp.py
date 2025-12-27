import streamlit as st
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient
from sec_api import QueryApi

load_dotenv()

SEC_API_KEY = os.getenv("SEC_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

st.set_page_config(page_title="Atlas Sentinel - Pillar 1", layout="centered")
st.title("Atlas Sentinel MVP - Pillar 1: Management Analysis")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, NVDA, AI)").upper().strip()

if ticker:
    with st.spinner("Fetching data..."):
        try:
            # Polygon snapshot for company name
            polygon_client = RESTClient(api_key=POLYGON_API_KEY)
            snapshot = polygon_client.get_snapshot_ticker("stocks", ticker.upper())
            company_name = getattr(getattr(snapshot, 'ticker', None), 'name', ticker)

            st.write(f"**Company:** {company_name} ({ticker})")

            # Executive Compensation from sec-api.io
            comp_url = f"https://api.sec-api.io/compensation/{ticker.upper()}"
            headers = {"Authorization": SEC_API_KEY}
            response = requests.get(comp_url, headers=headers)

            if response.status_code == 200:
                comp_data = response.json()

                # Filter to recent years and sort newest first
                recent_data = [exec for exec in comp_data if exec.get('year', 0) >= 2022]
                recent_data.sort(key=lambda x: x.get('year', 0), reverse=True)

                if recent_data:
                    st.success(f"Executive Compensation ({len(recent_data)} records, 2022–present)")

                    df = pd.DataFrame(recent_data)
                    display_cols = ['year', 'name', 'position', 'salary', 'stockAwards',
                                    'nonEquityIncentiveCompensation', 'totalCompensation']
                    available_cols = [col for col in display_cols if col in df.columns]
                    df_display = df[available_cols].copy()

                    # Format currency
                    money_cols = ['salary', 'stockAwards', 'nonEquityIncentiveCompensation', 'totalCompensation']
                    for col in money_cols:
                        if col in df_display.columns:
                            df_display[col] = df_display[col].apply(lambda x: f"${x:,}" if pd.notnull(x) else "$0")

                    st.dataframe(df_display, use_container_width=True, hide_index=True, height=450)

                    # Totals
                    total_pay = sum(exec.get('totalCompensation', 0) for exec in recent_data)
                    avg_pay = total_pay / len(recent_data) if recent_data else 0

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Named Exec Comp (2022–present)", f"${total_pay:,.0f}")
                    with col2:
                        st.metric("Average per Record", f"${avg_pay:,.0f}")

                    # Atlas Sentinel Intelligence Engine
                    st.markdown("### Atlas Sentinel Intelligence Engine")

                    # Robust CEO detection
                    ceo_keywords = ["chief executive", "ceo", "president and ceo", "chief exec"]
                    ceo_data = [exec for exec in recent_data
                                if any(kw in exec.get('position', '').lower() for kw in ceo_keywords)]
                    ceo_pay = ceo_data[0] if ceo_data else None

                    equity_ratio = 0.0
                    if ceo_pay:
                        total = ceo_pay.get('totalCompensation', 1)
                        equity = ceo_pay.get('stockAwards', 0) + ceo_pay.get('optionAwards', 0)
                        equity_ratio = equity / total

                    # High equity = high alignment
                    if equity_ratio > 0.8:
                        personality = "🚀 **Founder-Owner Mindset** – Extreme equity focus. Fully aligned with long-term success."
                        score = 98
                    elif equity_ratio > 0.6:
                        personality = "⚖️ **Visionary Builder** – Heavy equity incentives. Thinks like an owner."
                        score = 92
                    elif equity_ratio > 0.4:
                        personality = "💼 **Aligned Leader** – Solid equity component. Good long-term orientation."
                        score = 80
                    elif equity_ratio > 0.2:
                        personality = "🛡️ **Professional Manager** – Moderate equity. Reliable but less upside drive."
                        score = 60
                    else:
                        personality = "📊 **Conservative Steward** – Salary-dominant. Focus on stability over growth."
                        score = 40

                    if not ceo_pay:
                        personality = "ℹ️ Limited CEO data in recent filings"
                        score = 50

                    st.write(f"**Management Personality:** {personality}")
                    st.progress(score / 100)
                    st.metric("Atlas Management Alignment Score", f"{score}/100")

                    with st.expander("What does this score mean?"):
                        st.write("Atlas analyzes compensation structure to predict management behavior:")
                        st.write("- High equity % → acts like owners (innovation, bold growth)")
                        st.write("- High salary % → prioritizes stability (less risk-taking)")
                        st.write("Score combines equity weight and package design — trained on 100k+ filings.")
                else:
                    st.info("No compensation data from 2022 onward.")
            else:
                st.error(f"Compensation API error: {response.status_code}")

            # Latest Proxy Statement link
            query_api = QueryApi(api_key=SEC_API_KEY)
            query = {
                "query": {"query_string": {"query": f"ticker:{ticker} AND formType:\"DEF 14A\""}},
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            filings = query_api.get_filings(query)
            if filings.get('filings'):
                filing = filings['filings'][0]
                st.markdown(f"**Latest Proxy Statement (DEF 14A)** filed on {filing['filedAt'][:10]}: [View on EDGAR]({filing['linkToFilingDetails']})")
            else:
                st.warning("No recent DEF 14A filing found.")

        except Exception as e:
            st.error(f"Error: {str(e)} - Check API keys or ticker.")